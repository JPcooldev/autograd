# Optimizer internals

How gradients get from a backward pass into a parameter update in this library. Constructors: [reference.md](reference.md). The math of each optimizer is in [theory.md](theory.md).

## Training step

A typical iteration is:

1. **Forward.** Parameters (leaf tensors with `requires_grad = true`) participate in ops. The graph stores aliases of those leaves, not new copies of their data.
2. **Backward.** `loss.backward()` walks the graph and writes into each leaf’s gradient buffer.
3. **`optimizer.step()`.** Each optimizer reads `param->grad()` and writes the new weights in place into `param->data()`.
4. **`optimizer.zero_grad()`.** Gradient buffers are cleared so the next backward starts from zero. Optimizer state (`m`, `v`, step counters) is left alone.

```text
parameters (leaves)          graph (aliases share GradStorage)
        |                              ^
        |  forward ops                 |
        v                              |
      loss  ---- backward ---->  accumulate_grad
        |                              |
        v                              v
  optimizer.step()  <-----  param->grad()  (GradStorage::tensor)
        |
        v
  optimizer.zero_grad()  ---->  GradStorage::tensor = nullptr
```

`Module::parameters()` collects the same leaf pointers the optimizer holds. Either `module.zero_grad()` or `optimizer.zero_grad()` clears those buffers.

## Gradient storage on a tensor

Only **leaf** tensors that require grad own a `GradStorage` object. It is allocated at construction so later aliases can share it.

```text
Tensor (leaf, requires_grad)
  data_ptr_        -->  weight buffer  (what step() updates)
  grad_fn_         =    nullptr         (leaves are not op outputs)
  grad_storage_    -->  GradStorage
                          tensor  = nullptr until first accumulate_grad()
                                  = gradient Tensor after backward
```

- **Non-leaves** (op results) have a `grad_fn` and `grad_storage_ == nullptr`. Their incoming gradient lives only in the engine’s accumulator map for the duration of that backward.
- **No-grad tensors** also have `grad_storage_ == nullptr`. The engine does not write a gradient for them.

`grad()` returns `GradStorage::tensor.get()`, or `nullptr` if storage is missing or has not been written yet.

## How gradients are accumulated

On forward, each `Node` saves `Tensor::alias(x)` for its inputs. An alias copies the `shared_ptr` to `GradStorage`, so a write on the alias is visible on the original parameter.

`run_backward`:

1. Topologically sorts nodes from the loss toward the leaves.
2. For each node, calls `apply(node_grad)` to get per-input gradients.
3. If the next edge is another node, adds that gradient into the engine’s per-node accumulator.
4. If the next edge is `nullptr` **and** the saved input `requires_grad()`, this is a leaf: `saved_tensors[i].accumulate_grad(input_grads[i])`.

`accumulate_grad`:

- First call: allocates `GradStorage::tensor` as a no-grad copy of the incoming gradient.
- Later calls (second backward without `zero_grad`, or several paths to the same leaf): **adds in place**, `dst[i] += src[i]`.

So `param.grad()` is the sum of all leaf-bound contributions from the last backward (or from several backwards if you never cleared it).

## What `step()` does

`Optimizer` holds `std::vector<Tensor<T>*>` — raw pointers to user-owned (or module-owned) tensors. Those tensors must outlive the optimizer.

For each pointer, `step()`:

1. Skips if `!requires_grad()`.
2. Skips if `grad()` is `nullptr` (no backward since the last `zero_grad`, or the parameter was unused). **Adam / AdamW do not increment that parameter’s step counter** in this case, so bias correction stays aligned with real updates.
3. Otherwise reads `grad()->data()` and writes `param->data()` in place.

Weight decay is always folded into the update as an extra multiply-add (`g + wd · θ` for SGD/Adam, `lr · wd · θ` for AdamW). When `wd = 0` that term is zero; there is no separate kernel.

SGD has no extra state. Adam and AdamW keep, per parameter pointer:

- first moment `m`
- second moment `v`
- integer `step` (for bias correction)

That state is allocated in the optimizer constructor from each parameter’s current `data().size()` and lives until the optimizer is destroyed.

## Lifetimes

| Object | Created | Cleared / destroyed |
| --- | --- | --- |
| Parameter `data_ptr_` | Tensor construction | When the last tensor sharing the buffer dies |
| `GradStorage` (the box) | Leaf construction | With the leaf (and any aliases still holding the `shared_ptr`) |
| `GradStorage::tensor` (the gradient) | First `accumulate_grad` after a clear | `zero_grad()` sets it to `nullptr`; the box stays so the next forward’s aliases still share it |
| Engine accumulators | One `run_backward` | End of that call (stack / map locals) |
| Graph nodes / saved aliases | Forward ops | When nothing holds `grad_fn` / `saved_tensors` anymore |
| Adam `m`, `v`, `step` | Optimizer constructor | Optimizer destructor; **not** touched by `zero_grad()` |

`zero_grad()` must drop the gradient tensor but keep `GradStorage` itself. If the box were freed, a new forward would `alias()` a null `grad_storage_` and leaf gradients would no longer land on the original parameter.

In-place `step()` on a leaf is intentional: after backward, the weights are the values to update, not intermediates in the graph. Mutating a leaf **between** forward and backward would make saved aliases see the mutated values; this library does not version-check that (PyTorch would error).

## Who owns the pointers

```text
Module / user stack
  Linear.weight, Linear.bias, ...     // Tensor objects
        ^
        | raw pointers
Optimizer.parameters_
  Adam.state_[param*] -> { m, v, step }
```

The optimizer never copies parameter storage. If you move or destroy a parameter while an optimizer still points at it, the next `step()` is undefined. Build the optimizer from `model.parameters()` (or an explicit list) after the tensors exist, and keep both alive for the training run.
