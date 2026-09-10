# Operations

Free functions in the `ops::` namespace. Headers are in `src/ops/`.

Elementwise, shape, reduction, and linalg ops are also `Tensor` methods (`x.add(y)` ≡ `ops::add(x, y)`). **Not** Tensor methods: `cat`, `embedding`, conv, pool, dropout, and loss.

`#include "autograd/autograd.h"` pulls elementwise, shape, embedding, mixed-dtype, reduction, linalg, conv, and loss. Pool and dropout need their own headers (or a layer header that includes them):

```cpp
#include "ops/pool_ops.h"
#include "ops/dropout_ops.h"
```

When gradient tracking is active (`autograd::is_grad_enabled()`) and at least one input has `requires_grad = true`, the operation creates a backward node and attaches it to the output's `grad_fn`.

---

## Elementwise operations

**Header:** `src/ops/elementwise_ops.h`

All binary elementwise ops require inputs with **identical shapes**. There is no implicit broadcasting — use `ops::broadcast_to` first if shapes differ.

Elementwise ops loop `data()[i]` densely. Non-contiguous inputs are **packed at the start of the kernel** (`contiguous()`; no-op if already packed with `offset == 0`). Shape ops such as `transpose` and `broadcast_to` stay views.

That is preferred over a strided wrapper around the buffer (no extra Storage object; `data()` stays the raw `vector`) and over putting the stride formula in every kernel (a gather loop that usually will not auto-vectorize). Sequential `data()[i]` after a pack is what `-O2` can turn into SIMD. `matmul` is the exception: it already walks strides. See [tensor/mechanism.md](tensor/mechanism.md#why-dense-kernels-pack-at-the-door).

### Binary ops

| Function | Forward | Gradient (x) | Gradient (y) |
|---|---|---|---|
| `ops::add(x, y)` | `x + y` | `grad` | `grad` |
| `ops::subtract(x, y)` | `x - y` | `grad` | `-grad` |
| `ops::multiply(x, y)` | `x * y` | `y * grad` | `x * grad` |
| `ops::divide(x, y)` | `x / y` | `(1/y) * grad` | `(-x/y²) * grad` |

```cpp
Tensor<float> a({3}, {1.f, 2.f, 3.f}, true);
Tensor<float> b({3}, {4.f, 5.f, 6.f}, true);

auto c = ops::add(a, b);           // [5, 7, 9]
auto d = ops::multiply(a, b);      // [4, 10, 18]
```

### Unary ops

| Function | Forward | Gradient |
|---|---|---|
| `ops::neg(x)` | `-x` | `-grad` |
| `ops::abs(x)` | `\|x\|` | `sign(x) * grad` |
| `ops::sqrt(x)` | `√x` | `(1 / (2√x)) * grad` |
| `ops::exp(x)` | `eˣ` | `exp(x) * grad` |
| `ops::log(x)` | `ln(x)` | `(1/x) * grad` — throws `std::runtime_error` if any element is `≤ 0` |
| `ops::sin(x)` | `sin(x)` | `cos(x) * grad` |
| `ops::cos(x)` | `cos(x)` | `-sin(x) * grad` |
| `ops::tan(x)` | `tan(x)` | `(1/cos²(x)) * grad` |
| `ops::sinh(x)` | `sinh(x)` | `cosh(x) * grad` |
| `ops::cosh(x)` | `cosh(x)` | `sinh(x) * grad` |
| `ops::tanh(x)` | `tanh(x)` | `(1 − tanh²(x)) * grad` |
| `ops::sigmoid(x)` | `1/(1+e⁻ˣ)` | `σ(x)(1−σ(x)) * grad` |
| `ops::relu(x)` | `max(0, x)` | `(x > 0) * grad` |
| `ops::silu(x)` | `x · σ(x)` | `(σ(x) + x·σ(x)·(1−σ(x))) * grad` |
| `ops::gelu(x)` | tanh GELU approx | `GELUBackward` on the tanh form |

`gelu` uses \(0.5\, x\, (1 + \tanh(\sqrt{2/\pi}\,(x + 0.044715\, x^3)))\).

`ops::divide` throws `std::runtime_error` on a zero denominator.

```cpp
auto y = ops::sigmoid(x);
auto z = ops::relu(x);
auto s = ops::silu(x);
```

### Power

```cpp
ops::power(x, exponent)   // x ^ exponent, scalar exponent only
```

| Forward | Gradient (x) |
|---|---|
| `xⁿ` | `n * x^(n-1) * grad` |

The exponent is a scalar of type `T`, not a tensor.

---

## Shape operations

**Header:** `src/ops/shape_ops.h`

### `transpose`

```cpp
ops::transpose(x, dim0, dim1)   // swap two dimensions
x.transpose(dim0, dim1)

x.transpose()                   // rank-2 only: swap both dims
```

Returns a **view** (no data copy). Strides are swapped; underlying storage is shared.

Gradient: applies the same transpose in reverse (`TransposeBackward` calls `transpose(grad, dim0, dim1)`).

```cpp
Tensor<float> t({3, 4}, data, true);
auto t_T = t.transpose(0, 1);   // shape {4, 3}
```

### `reshape`

```cpp
ops::reshape(x, new_shape)
x.reshape(new_shape)
```

Returns a view with new shape and contiguous strides. Total element count must be preserved.

If the source is not contiguous, `reshape` packs it first (copy), then views. `view` never copies and throws if `!x.is_contiguous()`.

Gradient: reshapes incoming gradient back to the original shape (`ReshapeBackward`).

```cpp
Tensor<float> t({2, 6}, data, true);
auto r = t.reshape({3, 4});
```

### `view`

```cpp
ops::view(x, new_shape)
x.view(new_shape)
```

Alias for `reshape` **only when the source is already contiguous**. Throws `std::invalid_argument` if the tensor is not contiguous (never copies).

### `broadcast_to`

```cpp
ops::broadcast_to(x, target_shape)
x.broadcast_to(target_shape)
```

Expands dimensions by setting stride = 0 on broadcast axes. Follows NumPy broadcasting rules: shapes are right-aligned; a dimension of size 1 in the source can be expanded to any target size.

Does **not** copy data.

Gradient: sums the incoming gradient along all broadcast dimensions to recover the original shape (`BroadcastToBackward` → `sum_to`).

```cpp
Tensor<float> col({3, 1}, data, true);
auto expanded = col.broadcast_to({3, 4});   // {3, 4}, stride (1, 0)
```

### `flatten`

```cpp
ops::flatten(x, start_dim, end_dim)   // no defaults
x.flatten(start_dim=0, end_dim=-1)
```

Collapses dimensions `[start_dim, end_dim]` inclusive into a single dimension.

Packs first if the source is not contiguous.

Gradient: reshapes incoming gradient back to the original shape (`FlattenBackward`).

```cpp
Tensor<float> t({2, 3, 4}, data, true);
auto f = t.flatten(1, 2);   // shape {2, 12}
```

### `squeeze`

```cpp
ops::squeeze(x)        // remove all size-1 dimensions
ops::squeeze(x, dim)   // remove size-1 dimension at dim (no-op if size != 1)
x.squeeze()
x.squeeze(dim)
```

Returns a view. When `squeeze(dim)` is called and the dimension is not size 1, the returned tensor has `grad_fn = nullptr` (no gradient linkage). Always use `squeeze()` (no-arg) or ensure the dimension is actually size 1 when gradient flow is required.

Gradient: `SqueezeBackward` reshapes the gradient back to the pre-squeeze shape.

### `unsqueeze`

```cpp
ops::unsqueeze(x, dim)
x.unsqueeze(dim)
```

Inserts a size-1 dimension at position `dim`. Returns a view.

Gradient: `UnsqueezeBackward` squeezes the gradient at the same `dim`.

```cpp
Tensor<float> t({3, 4}, data, true);
auto u = t.unsqueeze(0);   // shape {1, 3, 4}
```

### `broadcast_shapes` (utility)

```cpp
std::vector<int64_t> ops::broadcast_shapes(
    const std::vector<int64_t>& a,
    const std::vector<int64_t>& b
);
```

Returns the broadcasted output shape following NumPy rules. Throws if shapes are incompatible.

### `contiguous`

```cpp
ops::contiguous(x)
x.contiguous()
```

Packs logical order into a dense row-major buffer with `offset == 0`. If `x` is already packed that way, returns `x` (same handle, no extra node). Otherwise gathers through strides and attaches `ContiguousBackward` (identity on values).

### `narrow`

```cpp
ops::narrow(x, dim, start, length)
x.narrow(dim, start, length)
```

View of a slice along `dim`: new size `length` starting at `start`. Same strides; offset moves by `start * stride[dim]`. Throws if `x` is scalar or the window is out of range. Negative `dim` is allowed.

Gradient: `NarrowBackward` writes the incoming gradient into a zeros-like tensor of the original shape at that window (scatter).

```cpp
Tensor<float> t({3, 4}, data, true);
auto mid = t.narrow(0, 1, 1);   // shape {1, 4}, view
```

### `cat`

```cpp
ops::cat(tensors, dim)
ops::cat({a, b, c}, dim)        // initializer_list overload
```

**Not** a Tensor method. Concatenates along `dim` (negative indices allowed). Inputs must share rank and match on every axis except `dim`. Scalars cannot be concatenated. Always allocates a packed result.

Gradient: `CatBackward` splits the incoming gradient and scatters each slice back to the corresponding input.

```cpp
auto a = Tensor<float>::ones({2, 3});
auto b = Tensor<float>::zeros({1, 3});
auto c = ops::cat({a, b}, 0);   // {3, 3}
```

---

## Reduction operations

**Header:** `src/ops/reduction_ops.h`

### `sum`

```cpp
ops::sum(x)          // global sum → scalar {}
ops::sum(x, dim)     // reduce along dim → removes that dimension

x.sum()
x.sum(dim)
```

Global `sum` produces a rank-0 (scalar) tensor.

`sum(dim)` / `mean(dim)` pack first (`contiguous()`), then reduce the axis as a dense outer × reduce × inner add. Non-contiguous inputs are packed at the start of the kernel.

Gradients:

- `SumBackward`: broadcasts the scalar gradient back to the original shape.
- `SumDimBackward`: broadcasts the gradient back along the reduced dimension.

### `mean`

```cpp
ops::mean(x)
ops::mean(x, dim)

x.mean()
x.mean(dim)
```

Equivalent to `sum / count`. Gradient scales by `1 / numel` (global) or `1 / size(dim)` (dim variant). Dim kernels use the same packed outer × reduce × inner layout as `sum(dim)`.

### `max` / `min`

```cpp
ops::max(x)         // global maximum → scalar
ops::max(x, dim)    // reduce along dim → removes that dimension
ops::min(x)
ops::min(x, dim)

x.max()
x.max(dim)
x.min()
x.min(dim)
```

Ties keep the first index along the reduced axis (or the first flat index for the global variants). Empty `dim` throws. Dim kernels pack, then compare contiguous inner-length slabs.

Gradients: one-hot at the argmax/argmin position (`MaxBackward` / `MinBackward` globally; `MaxDimBackward` / `MinDimBackward` scatter per output slot).

### `softmax`

```cpp
ops::softmax(x, dim)
x.softmax(dim)
```

Same shape as input. Each slice along `dim` sums to 1. Uses max-subtraction for stability, then the packed outer × reduce × inner layout (dim is kept in the output). Empty `dim` throws.

Gradient: `dx = s * (g - sum(g * s, dim))` per slice (`SoftmaxBackward`).

---

## Linear algebra

**Header:** `src/ops/linalg_ops.h`

`matmul` is **stride-safe**: it uses the full `offset + sum(idx * stride)` formula and therefore works correctly on transposed or non-contiguous views. `dot` packs first, then uses a dense `data()[i]` loop.

### `dot`

```cpp
ops::dot(a, b)
a.dot(b)
```

Requires rank-1 inputs of equal length. Returns a scalar `{}`.

Gradients: `DotBackward` — `grad_a = b * scalar_grad`, `grad_b = a * scalar_grad`.

```cpp
Tensor<float> u({4}, {1, 2, 3, 4}, true);
Tensor<float> v({4}, {5, 6, 7, 8}, true);
auto s = ops::dot(u, v);   // scalar {70}
```

### `matmul`

```cpp
ops::matmul(a, b)
a.matmul(b)
```

Both operands must have rank ≥ 2. The last two dimensions are the matrix axes: `A[..., M, K] @ B[..., K, N] → C[..., M, N]`. Leading dimensions are batch axes and are broadcast together (NumPy / PyTorch rules). `A`'s last dim must equal `B`'s second-to-last dim.

Internally `MatmulBackward` stores contiguous copies of `A` and `B` so the backward loops have guaranteed layout. Gradients over a broadcast batch axis are summed back to the original shape.

Gradients (per batch):

```
grad_A = grad_C @ B^T
grad_B = A^T @ grad_C
```

```cpp
Tensor<float> A({2, 3}, data_a, true);
Tensor<float> B({3, 4}, data_b, true);
auto C = ops::matmul(A, B);   // shape {2, 4}

Tensor<float> batched({8, 2, 3}, data, true);
auto D = ops::matmul(batched, B);   // shape {8, 2, 4}  — B broadcasts over the batch
```

---

## Embedding

**Header:** `src/ops/embedding_ops.h`

```cpp
ops::embedding(weight, indices)
```

`weight` is a rank-2 table `{V, D}`. `indices` is an integer tensor (`int32_t` or `int64_t`) of any rank. Output shape is `indices.shape + {D}`.

Gradient: scatter-add of the output gradient into the selected rows of `weight`. Repeated indices accumulate. Indices themselves are not differentiated.

```cpp
Tensor<float> table({4, 8}, true);          // V=4, D=8
Tensor<int32_t> idx({2, 3}, token_ids, false);
auto y = ops::embedding(table, idx);        // {2, 3, 8}
```

The layer wrapper is `nn::Embedding` (see [layers/reference.md](layers/reference.md)).

---

## Convolution

**Header:** `src/ops/conv_ops.h`

NCHW: 1-D `(N, C, L)`, 2-D `(N, C, H, W)`, 3-D `(N, C, D, H, W)`. `stride` / `padding` / `dilation` are one `int64_t` applied to every spatial axis. No `groups`, no `padding='same'`. Inputs are packed first.

Weight layout: conv `{C_out, C_in, *K}`; transpose `{C_in, C_out, *K}`. Optional `bias` is `const Tensor<T>*` of shape `{C_out}` (or `nullptr`).

```cpp
ops::conv1d(input, weight, bias=nullptr, stride=1, padding=0, dilation=1)
ops::conv2d(...)   // (N, C, H, W)
ops::conv3d(...)   // (N, C, D, H, W)

ops::conv_transpose1d(input, weight, bias=nullptr, stride=1, padding=0,
                      dilation=1, output_padding=0)
ops::conv_transpose2d(...)
ops::conv_transpose3d(...)
```

`conv_nd` / `conv_transpose_nd` are the shared kernels (1–3 spatial dims). Rank is taken from the input.

Output size:

```text
out = (in + 2·pad - dilation·(k-1) - 1) / stride + 1
transpose: (in-1)·stride - 2·pad + dilation·(k-1) + output_padding + 1
```

Gradient: `ConvBackward` / `ConvTransposeBackward` — `grad_input` is the transpose-conv of `grad_output` with `W`; `grad_weight` correlates input with `grad_output`; `grad_bias` sums over batch and spatial axes.

```cpp
Tensor<float> x({1, 1, 8, 8}, true);
Tensor<float> w({4, 1, 3, 3}, true);
auto y = ops::conv2d(x, w, nullptr, /*stride=*/1, /*padding=*/1);
```

Layer wrappers: `nn::Conv2d`, `nn::ConvTranspose2d`, … — they always pass `output_padding=0` on transpose.

---

## Pooling

**Header:** `src/ops/pool_ops.h` (not in `autograd.h`)

Same NCHW ranks as conv. `stride = -1` means `stride = kernel`. Avg pool divides by the full kernel volume (zeros from padding count; `count_include_pad=true`). Max pool windows that lie entirely in padding write `0`. Ties keep the first index in the window.

```cpp
ops::max_pool1d(x, kernel, stride=-1, padding=0)
ops::max_pool2d(...)
ops::max_pool3d(...)
ops::avg_pool1d(x, kernel, stride=-1, padding=0)
ops::avg_pool2d(...)
ops::avg_pool3d(...)

ops::global_avg_pool(x)   // (N, C, *spatial) → (N, C); rank ≥ 3
ops::global_max_pool(x)
```

`max_pool_nd` / `avg_pool_nd` are the shared kernels. Spatial output size uses the conv formula with `dilation=1`.

Gradient: `MaxPoolBackward` scatters to saved argmax; `AvgPoolBackward` spreads `grad / k^D` over each window.

---

## Dropout

**Header:** `src/ops/dropout_ops.h` (not in `autograd.h`)

Inverted dropout. **Not** a Tensor method.

```cpp
ops::dropout(x, p, training, seed=nullopt)
```

`p` is the drop probability, in `[0, 1)`. If `!training` or `p == 0`, returns `x` unchanged (no node). Otherwise each element is kept with probability `1-p` and scaled by `1/(1-p)`. Optional `seed` makes the mask repeatable.

Gradient: `DropoutBackward` multiplies by the saved mask (zeros and the keep-scale). There is no public `bernoulli` factory.

```cpp
auto y = ops::dropout(x, 0.5, /*training=*/true, /*seed=*/1);
```

The layer `nn::Dropout` passes `training()`.

---

## Loss

**Header:** `src/ops/loss_ops.h`

All return a scalar `{}`. Input and target must have the **same shape**. There is no `nn::NLLLoss` layer — use `ops::nll_loss` directly. The other ops have thin `nn::*Loss` wrappers that always use mean reduction.

| Function | Formula | Notes |
|---|---|---|
| `l1_loss(input, target, reduction=true)` | mean or sum of `\|input-target\|` | MAE |
| `l2_loss(input, target, reduction=true)` | mean or sum of `(input-target)²` | |
| `mse_loss(...)` | alias of `l2_loss` | |
| `nll_loss(input, target, dim=-1)` | `-(1/N_batch) Σ target · input` | `input` = **log-probs**; **soft** labels |
| `cross_entropy_loss(input, target, dim=-1)` | fused log-softmax + NLL | `input` = logits; **soft** labels, same shape |
| `bce_loss(input, target)` | mean BCE | `input` = probabilities, clamped |
| `bce_with_logits_loss(input, target)` | stable sigmoid+BCE | `input` = logits |
| `kl_div_loss(input, target)` | `mean(target · (log(target) - input))` | `input` = log-probs, `target` = probs |

`N_batch` for NLL / cross-entropy is `numel / size(dim)` (number of class-axis slices). Class indices are **not** accepted — one-hot or probability targets.

```cpp
auto logits = model.forward(x);                 // {N, C}
auto loss = ops::cross_entropy_loss(logits, one_hot);
loss.backward();
```

---

## Mixed-dtype operations

**Header:** `src/ops/mixed_dtype_ops.h`

When two tensors have different element types the standard `ops::` functions cannot be called directly (they require `Tensor<T>` + `Tensor<T>`). The mixed-dtype header provides overloads for every two-tensor `ops::` entry point:

```cpp
ops::add / subtract / multiply / divide
ops::dot / matmul
ops::l1_loss / l2_loss / mse_loss / nll_loss / cross_entropy_loss
ops::bce_loss / bce_with_logits_loss / kl_div_loss
ops::conv1d / conv2d / conv3d
ops::conv_transpose1d / conv_transpose2d / conv_transpose3d
```

The result type is `std::common_type_t<T1, T2>`. Both operands are first converted with `.to<R>()` (detaches them from the graph), and the result is a `Tensor<R>`. Extra arguments (`reduction`, `dim`, conv stride/padding) are forwarded unchanged. Conv `bias`, if present, must already be `Tensor<R>` (caller `.to<R>()` otherwise).

Not mixed-dtype: unary ops, reductions, pool, dropout, views, `power` (scalar exponent), `cat` (`vector<Tensor<T>>` cannot hold mixed types — `.to<R>()` first), `Tensor<T>` members, and `Module<T>`.

**Cross-dtype autograd is not first-class.** The graph is built on the promoted `Tensor<R>` copies, not the original typed tensors. Gradients flow back through the promoted copies only.

Type promotion table:

| T1 | T2 | Result |
|---|---|---|
| `float` | `double` | `double` |
| `float` | `int32_t` | `float` |
| `float` | `int64_t` | `float`* |
| `double` | `int32_t` | `double` |
| `double` | `int64_t` | `double` |
| `int32_t` | `int64_t` | `int64_t` |

*`int64_t → float` can lose precision for large integers.

```cpp
Tensor<float>  f({3}, {1.f, 2.f, 3.f});
Tensor<double> d({3}, {0.1, 0.2, 0.3});

auto r = ops::add(f, d);   // Tensor<double>
```
