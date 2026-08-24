# Layer internals

How `nn::Layer` / `nn::Module` own parameters, how those leaves stay alive for `backward()`, and how the thicker layers (conv, RNN, dropout, batch-norm) are implemented. The per-layer API is in [reference.md](reference.md). Optimizer consumption of `.grad()` is in [../optim/mechanism.md](../optim/mechanism.md).

## Layer vs Module vs leaf parameters

A **layer** is a `nn::Layer<T>` with its own `forward` signature and a `parameters()` list of **leaf** tensors (`requires_grad == true`, `grad_fn == nullptr`). There is no virtual `forward` on the base: Python `nn.Module` does the same, and it is the only way to express loss `(input, target)`, RNN `(output, h_n)`, and mutating Dropout / BatchNorm.

A **module** is a `nn::Module<T>` — a layer that also holds named sub-layers. Because C++ has no `__setattr__` hook, you `register_module` (and optionally `register_parameter`) in the constructor. `parameters()` walks `own_params_` then every registered child, depth-first.

```text
Module
  own_params_     raw Tensor* registered on this module
  submodules_     raw Layer*  (must outlive the module — store them as members)
        |
        v
  parameters() = own_params_ + child.parameters() + ...
```

`Linear` / `Conv*` / RNN cells keep `weight` and `bias` as **public members** that are those leaves. Forward ops receive `Tensor::alias` of them (see below), so the graph does not copy the buffers.

## `Tensor::alias` and graph lifetime

`Node` constructors store `Tensor::alias(x)` in `saved_tensors`. For a leaf this copies the `shared_ptr` to `GradStorage` and the data buffer, not the numbers. `run_backward` writes `accumulate_grad` on that alias; the layer's `weight.grad()` sees it.

The contract is the same as PyTorch `SavedVariable` for leaves: **the layer must outlive the backward pass**. A typical step is:

1. `y = layer.forward(x)` — graph holds aliases of `layer.weight`.
2. `loss.backward()` — fills `layer.weight.grad()`.
3. `opt.step()` / `zero_grad()` — the layer is still in scope.

If you destroy the layer (or a temporary `Linear` built in the `forward` expression) before `backward()`, the aliases dangle. Don't do that.

## `train()` / `eval()` recursion

`Layer` stores `training_ = true` by default. `train(mode)` sets it; `eval()` is `train(false)`.

`Module::train` sets its own flag **and** calls `train(mode)` on every registered submodule. That is how `model.eval()` turns off Dropout and switches BatchNorm to running stats without touching each child.

Layers that ignore the flag (Linear, Conv, Pool, RNN, most losses) still inherit it; it is harmless.

## Why `forward` is not virtual

A single `virtual Tensor forward(const Tensor&) const` cannot express:

| Layer | Signature |
| --- | --- |
| Linear, Conv, Pool, Dropout, Norm | `Tensor forward(const Tensor&)` |
| Dropout, BatchNorm | **non-const** `forward` (mask / running stats) |
| Losses | `Tensor forward(input, target)` |
| RNN / GRU | `pair<output, h_n>` |
| LSTM | `pair<output, pair<h_n, c_n>>` |

Each class defines the method it needs. There is no `override` on user `Module::forward` either.

## Convolution: NCHW window loops

Layouts are NCHW (1-D: `N, C, L`; 3-D: `N, C, D, H, W`). Weights are `{C_out, C_in, *k}` for conv and `{C_in, C_out, *k}` for transpose. `kernel` / `stride` / `padding` / `dilation` are one `int64_t` applied to every spatial axis. No `groups`, no `padding='same'`.

The kernel is an explicit nested window (educational, not im2col):

```text
out[n, oc, *p] = bias[oc] + Σ_ic Σ_k  in[n, ic, stride·p + dilation·k - padding] * W[oc, ic, *k]
```

Output size:

```text
out = (in + 2·pad - dilation·(k-1) - 1) / stride + 1
```

Transpose (scatter of the same geometry):

```text
out = (in-1)·stride - 2·pad + dilation·(k-1) + output_padding + 1
```

Backward:

- `grad_input` is the transpose-conv of `grad_output` with `W`
- `grad_weight` correlates `input` with `grad_output`
- `grad_bias` sums `grad_output` over batch and spatial axes

Inputs are `contiguous()`'d first; dense kernels assume `offset == 0`.

## RNN: unroll in T, alias the same leaves

`RNN` / `LSTM` / `GRU` loop over time. Each step is `matmul` + the usual nonlinearities (LSTM/GRU split packed gates with `narrow`). The **same** `weight_ih` / `weight_hh` leaves are passed into every step.

Because each `matmul` node aliases those leaves, `weight.grad()` after `output.sum().backward()` is the **sum over time** (and over the batch) of the local Jacobians — the same accumulation PyTorch gets from sharing `nn.Parameter` across the unroll.

Default layout is seq-first `(T, N, F)` like `torch.nn.RNN`. `batch_first=true` uses `(N, T, F)`. Bidirectional runs a second cell in reverse and `cat`s on the feature axis. `num_layers > 1` stacks; there is no inter-layer dropout (use `nn::Dropout` yourself).

`h0` default is zeros with `requires_grad=false`. LSTM also carries `c_n`.

## Dropout mask

Inverted dropout: in `training()`, each element is kept with probability `1-p` and **multiplied by `1/(1-p)`**. The mask is sampled inside `ops::dropout` (there is no public `bernoulli` factory) and saved on the backward node. `eval()` or `p == 0` returns the input unchanged (identity, no node).

## BatchNorm running stats vs affine parameters

`weight` / `bias` are leaves and appear in `parameters()`. `running_mean` / `running_var` are buffers (`requires_grad=false`) and **do not**.

In `training()`:

1. Compute batch mean/var over all axes except channel (dim 1).
2. Normalize with those stats + `eps`.
3. EMA: `running = (1-momentum)·running + momentum·batch`.

In `eval()`, normalize with `running_mean` / `running_var` only. Affine `y = γ ⊙ x̂ + β` is applied in both modes.
