# Operations

All operations are free functions inside the `ops::` namespace. Every operation is also available as an equivalent method on `tensor::Tensor<T>`. Headers are in `src/ops/`.

When gradient tracking is active (`autograd::is_grad_enabled()`) and at least one input has `requires_grad = true`, the operation automatically creates a backward node and attaches it to the output tensor's `grad_fn`. No extra wiring is required.

---

## Elementwise operations

**Header:** `src/ops/elementwise_ops.h`

All binary elementwise ops require inputs with **identical shapes**. There is no implicit broadcasting — use `ops::broadcast_to` first if shapes differ.

Elementwise ops read `data()[i]` sequentially and therefore require **contiguous** input tensors.

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
| `ops::exp(x)` | `eˣ` | `exp(x) * grad` |
| `ops::log(x)` | `ln(x)` | `(1/x) * grad` |
| `ops::sin(x)` | `sin(x)` | `cos(x) * grad` |
| `ops::cos(x)` | `cos(x)` | `-sin(x) * grad` |
| `ops::tan(x)` | `tan(x)` | `(1/cos²(x)) * grad` |
| `ops::sigmoid(x)` | `1/(1+e⁻ˣ)` | `σ(x)(1−σ(x)) * grad` |
| `ops::relu(x)` | `max(0, x)` | `(x > 0) * grad` |

```cpp
auto y = ops::sigmoid(x);
auto z = ops::relu(x);
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

**Requires contiguous input.** Throws `std::invalid_argument` if `!x.is_contiguous()` or if numel does not match.

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

Alias for `reshape`. Same constraints apply.

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
ops::flatten(x, start_dim, end_dim)
x.flatten(start_dim, end_dim)
```

Collapses dimensions `[start_dim, end_dim]` inclusive into a single dimension.

**Requires contiguous input.**

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

`sum(dim)` uses contiguous stride-formula indexing — **input must be contiguous** (or have contiguous logical layout along the reduced dimension).

Gradients:

- `SumBackward`: broadcasts the scalar gradient back to the original shape.
- `SumDimBackward`: broadcasts the 1-D gradient back along the reduced dimension.

### `mean`

```cpp
ops::mean(x)
ops::mean(x, dim)

x.mean()
x.mean(dim)
```

Equivalent to `sum / count`. Gradient scales by `1 / numel` (global) or `1 / size(dim)` (dim variant).

### `max` / `min`

```cpp
ops::max(x)    // global maximum → scalar
ops::min(x)    // global minimum → scalar

x.max()
x.min()
```

Only global variants are implemented. The index of the first maximum/minimum element is saved for the backward pass.

Gradients: one-hot at the argmax/argmin position (`MaxBackward` / `MinBackward`).

---

## Linear algebra

**Header:** `src/ops/linalg_ops.h`

`dot` and `matmul` are **stride-safe**: they use the full `offset + sum(idx * stride)` formula and therefore work correctly on transposed or non-contiguous views.

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

Requires rank-2 inputs; `a.shape()[1]` must equal `b.shape()[0]`.

Internally `MatmulBackward` stores contiguous copies of `A` and `B` (via `to_contiguous_2d`) so the backward loops have guaranteed layout.

Gradients (standard matrix calculus):

```
grad_A = grad_C @ B^T
grad_B = A^T @ grad_C
```

```cpp
Tensor<float> A({2, 3}, data_a, true);
Tensor<float> B({3, 4}, data_b, true);
auto C = ops::matmul(A, B);   // shape {2, 4}
```

---

## Mixed-dtype operations

**Header:** `src/ops/mixed_dtype_ops.h`

When two tensors have different element types the standard `ops::` functions cannot be called directly (they require `Tensor<T>` + `Tensor<T>`). The mixed-dtype header provides overloads:

```cpp
ops::add<T1, T2>(x, y)
ops::subtract<T1, T2>(x, y)
ops::multiply<T1, T2>(x, y)
ops::divide<T1, T2>(x, y)
```

The result type is `std::common_type_t<T1, T2>`. Both operands are first converted with `.to<R>()` (detaches them from the graph), and the result is a `Tensor<R>`.

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
