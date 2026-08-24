# Layer reference

Short cards for every `nn::` layer. Internals and the leaf/`alias` contract are in [mechanism.md](mechanism.md). Init numbers are PyTorch `reset_parameters`, not ad-hoc Gaussians.

Default layout for conv/pool is **NCHW**. RNN default is **seq-first** `(T, N, F)`.

---

## Identity

Pass-through. No parameters.

```cpp
Identity<T>()
Tensor forward(const Tensor& input) const  // same tensor
```

---

## Linear

Fully connected: \(y = x W^\top + b\) (bias optional).

- **Ctor:** `Linear(in_features, out_features, use_bias=true)`
- **Input:** `{in}` or `{N, in}` → `{out}` or `{N, out}`
- **Parameters:** `weight {out, in}`, optional `bias {out}`
- **Init:** `kaiming_uniform(W, a=√5)` = \(U(-1/\sqrt{\mathrm{fan\_in}},\, 1/\sqrt{\mathrm{fan\_in}})\); bias the same bound (not zeros)

---

## Conv1d / Conv2d / Conv3d

\(y = \mathrm{conv}(x, W) + b\). Single int `kernel`/`stride`/`padding`/`dilation` on every spatial axis. Zeros pad. No `groups`.

- **Ctor:** `Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, use_bias=true)`
- **Input:** `(N, C_in, *spatial)` → `(N, C_out, *out)` with \(out = (in + 2p - d(k-1) - 1)/s + 1\)
- **Parameters:** `weight {C_out, C_in, *k}`, optional `bias {C_out}`
- **Init:** same as Linear — Kaiming uniform \(a=\sqrt{5}\) on `weight`; bias \(U(-1/\sqrt{\mathrm{fan\_in}}, 1/\sqrt{\mathrm{fan\_in}})\) from `compute_fans(weight)`

---

## ConvTranspose1d / 2d / 3d

Scatter form of the same geometry. Weight layout `{C_in, C_out, *k}` (PyTorch). `output_padding` defaults to 0 (op argument, not a layer kwarg).

- **Ctor:** same as Conv*
- **Output size:** \((in-1)s - 2p + d(k-1) + \mathrm{output\_padding} + 1\)
- **Init:** `_ConvNd.reset_parameters` — same Kaiming + uniform bias; fans are taken from the stored weight, so transpose fan-in is correct

---

## MaxPool / AvgPool 1d/2d/3d

Windowed reduce. No parameters.

- **Ctor:** `MaxPool2d(kernel_size, stride=kernel_size, padding=0)`
- **Input:** `(N, C, *spatial)` → pooled spatial size (same formula as conv with `dilation=1`)
- Avg pool uses `count_include_pad=true` (divide by \(k^D\), including zeros from padding)
- Max pool backward scatters to the argmax (ties: first in window)

---

## GlobalMaxPool / GlobalAvgPool 1d/2d/3d

Reduce every spatial axis; keep `(N, C)`. No parameters. `forward` is the same helper for 1d/2d/3d (rank ≥ 3).

---

## RNN / LSTM / GRU

Unrolled recurrent stack. Output width is `hidden_size` (or `2·hidden` if bidirectional). **No** `output_size`.

- **Ctor:** `RNN(input_size, hidden_size, num_layers=1, use_bias=true, batch_first=false, bidirectional=false, nonlinearity="tanh")` — nonlinearity `"tanh"` | `"relu"` (RNN only)
- **Input:** `(T, N, input_size)` or, if `batch_first`, `(N, T, input_size)`
- **Output:** `forward(input)` / `forward(input, h0)` → `(output, h_n)`  
  LSTM: `(output, (h_n, c_n))`  
  `h_n` / `c_n` shape `(num_layers · num_directions, N, hidden_size)`
- **Parameters:** per layer, per direction: `weight_ih {G·H, in}`, `weight_hh {G·H, H}`, optional biases. \(G=1\) (RNN), \(4\) (LSTM, i/f/g/o), \(3\) (GRU, r/z/n)
- **Init:** every weight **and** bias \(U(-1/\sqrt{H}, 1/\sqrt{H})\) — not Kaiming, not zero bias

---

## LayerNorm

Last-axis affine LayerNorm. \(y = \frac{x-\mu}{\sqrt{\sigma^2+\varepsilon}} \odot \gamma + \beta\)

- **Ctor:** `LayerNorm(num_features, eps=1e-5)`
- **Input:** `(..., num_features)`
- **Parameters:** `weight` (γ) ones, `bias` (β) zeros

---

## RMSNorm

\(y = \frac{x}{\sqrt{\mathrm{mean}(x^2)+\varepsilon}} \odot \gamma\). No bias.

- **Ctor:** `RMSNorm(num_features, eps=1e-5)`
- **Parameters:** `weight` ones

---

## BatchNorm1d / 2d / 3d

Normalize over all axes except channel dim 1. `forward` is **non-const**.

- **Ctor:** `BatchNorm2d(num_features, eps=1e-5, momentum=0.1)`
- **Input:** 1d `(N, C)` or `(N, C, L)`; 2d `(N, C, H, W)`; 3d `(N, C, D, H, W)`
- **Parameters:** `weight` ones, `bias` zeros
- **Buffers (not in `parameters()`):** `running_mean` 0, `running_var` 1 (`requires_grad=false`)  
  Train: batch stats + `running = (1-m)·running + m·batch`. Eval: running stats only.

---

## Dropout

Inverted dropout. No parameters. `forward` is **non-const** (reads `training_`).

- **Ctor:** `Dropout(p=0.5)` with \(p \in [0, 1)\)
- **Train:** keep with prob \(1-p\), scale \(1/(1-p)\). **Eval** or \(p=0\): identity

---

## MSELoss / L1Loss / CrossEntropyLoss / BCELoss / BCEWithLogitsLoss / KLDivLoss

Thin wrappers around `ops::*`. No parameters. `forward(input, target)`.

| Class | Op | Notes vs PyTorch |
| --- | --- | --- |
| `MSELoss` | `mse_loss` | mean reduction |
| `L1Loss` | `l1_loss` | MAE, mean |
| `CrossEntropyLoss` | `cross_entropy_loss` | **soft labels**, same shape as logits (not class indices) |
| `BCELoss` | `bce_loss` | |
| `BCEWithLogitsLoss` | `bce_with_logits_loss` | |
| `KLDivLoss` | `kl_div_loss` | input = log-probs |

---

## Init table (PyTorch `reset_parameters`)

| Module | Weights | Bias / extra |
| --- | --- | --- |
| Linear, Conv*, ConvTranspose* | `kaiming_uniform(a=√5)` | \(U(-1/\sqrt{fan_{in}}, 1/\sqrt{fan_{in}})\) |
| RNN, LSTM, GRU | \(U(-1/\sqrt{H}, 1/\sqrt{H})\) on **all** parameters | same, including bias |
| LayerNorm | ones | zeros |
| RMSNorm | ones | — |
| BatchNorm* | ones | zeros; running mean 0, var 1 |
| Dropout, Pool, Loss, Identity | — | — |
