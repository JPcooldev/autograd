# Tensor methods

Public API of `tensor::Tensor<T>`. Storage and views are explained in [mechanism.md](mechanism.md).

`T` is `float`, `double`, `int32_t`, or `int64_t`. There is no bool dtype and **no `bool_()`**. Integer tensors never `requires_grad()`.

Include `src/autograd/autograd.h` for math, shape ops, and `backward()` (it pulls in `tensor_methods.tpp`). Construction and accessors live in `tensor.h` alone.

---

## Attributes

Private fields of `Tensor<T>`. Copying a tensor copies this metadata and **increments refcounts**; it does not clone the numbers.

| Field | Type | Role |
| --- | --- | --- |
| `shape_` | `vector<int64_t>` | Logical sizes of each axis. |
| `strides_` | `vector<int64_t>` | Step in the buffer along each axis, in **elements** (not bytes). |
| `offset_` | `int64_t` | Starting index into the buffer (`0` for owners). |
| `data_ptr_` | `shared_ptr<vector<T>>` | The only heap allocation for values. Shared by all views of this buffer. |
| `base_ptr_` | `shared_ptr<Tensor<T>>` | Non-null iff this object is a view. Keeps the source handle alive. |
| `dtype_` | `Dtype` | Runtime tag inferred from `T` (`Float32`, `Float64`, `Int32`, `Int64`). |
| `requires_grad_` | `bool` | Whether autograd tracks this tensor. Forced false for integer `T`. |
| `grad_fn_` | `shared_ptr<Node<T>>` | Backward node that produced this tensor, or `nullptr` for a leaf. |
| `grad_storage_` | `shared_ptr<GradStorage>` | Leaf-only box that holds the accumulated `.grad()`. `nullptr` on op outputs. |

`GradStorage` itself is a tiny struct: `{ shared_ptr<Tensor<T>> tensor; }`. The box is allocated at leaf construction; `tensor` stays `nullptr` until the first `accumulate_grad()`.

Logical index `(i0, i1, …)` maps to `data()[offset_ + i0*strides_[0] + i1*strides_[1] + …]`.

```cpp
tensor::Tensor<float> x({2, 3}, {1,2,3,4,5,6}, false);
// owner: shape {2,3}, strides {3,1}, offset 0, base_ptr_ == nullptr
auto t = x.transpose();
// view:  shape {3,2}, strides {1,3}, offset 0, base_ptr_ != nullptr
//        same data_ptr_ as x
```

---

## Constructors

Every constructor allocates a **new** packed buffer (`offset = 0`, contiguous strides). `requires_grad` defaults to `true` but is forced off for integer `T`.

| Method | Description |
| --- | --- |
| `Tensor(shape, requires_grad=true)` | Allocates packed storage and zero-fills it (`T{}`). |
| `Tensor(shape, fill, requires_grad)` | Allocates packed storage and fills every element with `fill`. |
| `Tensor(shape, vector, requires_grad)` | Copies a `std::vector` into a new buffer. Throws if `vector.size() != numel(shape)`. |
| `Tensor(shape, ptr, count, requires_grad)` | Copies `count` elements from a raw pointer. Throws if `count` does not match the shape, or if `ptr` is null when `count > 0`. |
| copy / move | Default. Copies or moves the metadata and `shared_ptr` handles (same buffer, handle semantics). |

```cpp
tensor::Tensor<float> a({2, 3});                       // 2x3 zeros, requires_grad
tensor::Tensor<float> b({2, 3}, 1.f, false);           // 2x3 ones, no grad
tensor::Tensor<float> c({2, 3}, {1,2,3,4,5,6}, true);  // copy from vector
float raw[] = {7, 8, 9};
tensor::Tensor<float> d({3}, raw, 3, false);
tensor::Tensor<float> e = c;                           // shares c's buffer
```

---

## Fill factories

Static constructors that allocate a new packed tensor. `*_like` copies **shape only**, not values or strides.

| Method | Description |
| --- | --- |
| `zeros(shape, requires_grad=true)` | New tensor filled with `0`. |
| `ones(shape, requires_grad=true)` | New tensor filled with `1`. |
| `full(shape, value, requires_grad=true)` | New tensor filled with `value`. |
| `zeros_like(other, requires_grad=true)` | `zeros(other.shape())`. |
| `ones_like(other, requires_grad=true)` | `ones(other.shape())`. |
| `full_like(other, value, requires_grad=true)` | `full(other.shape(), value)`. |

```cpp
auto z = tensor::Tensor<float>::zeros({2, 2}, false);
auto o = tensor::Tensor<float>::ones_like(z, false);
auto f = tensor::Tensor<float>::full({2}, 3.5f, false);
```

---

## Random and weight init

All of these allocate a **new** packed buffer. Optional `seed` makes draws repeatable (`mt19937_64`).

| Method | Description |
| --- | --- |
| `randint(shape, low, high, requires_grad=false, seed)` | Uniform integers in `[low, high)`. Throws if `low >= high`. |
| `randint_like(other, low, high, …)` | `randint` with `other.shape()`. |
| `randn(shape, requires_grad=true, seed)` | i.i.d. samples from `N(0, 1)`. |
| `randn_like(other, …)` | `randn` with `other.shape()`. |
| `random_gaussian(shape, mean, stddev, requires_grad=true, seed)` | i.i.d. samples from `N(mean, stddev)`. |
| `random_gaussian_like(other, mean, stddev, …)` | Same, shape taken from `other`. |
| `xavier_uniform(shape, gain=1, …)` | Glorot uniform: `U[-limit, limit]` with `limit = gain * sqrt(6 / (fan_in + fan_out))`. |
| `xavier_normal(shape, gain=1, …)` | Glorot normal: `N(0, gain * sqrt(2 / (fan_in + fan_out)))`. |
| `kaiming_uniform(shape, negative_slope=0, fan_mode="fan_in", …)` | He uniform: `U[-bound, bound]` with `bound = sqrt(3) * gain / sqrt(fan)`. |
| `kaiming_normal(shape, negative_slope=0, fan_mode="fan_in", …)` | He normal: `N(0, gain / sqrt(fan))`. |
| `compute_fans(shape)` | Returns `{fan_in, fan_out}`. Rank 2: `fan_in = shape[1]`, `fan_out = shape[0]`. Extra dims multiply both (conv weights). Rank 1: both equal `shape[0]`. |

`randn` / `random_gaussian` / Xavier / Kaiming are meant for floating `T`. Kaiming `gain` is `sqrt(2 / (1 + negative_slope²))`. `fan_mode` must be `"fan_in"` or `"fan_out"`.

```cpp
auto w = tensor::Tensor<float>::xavier_uniform({4, 8}, 1.0, true, /*seed=*/1);
auto n = tensor::Tensor<float>::randn({2, 2}, false, 42);
auto i = tensor::Tensor<int32_t>::randint({3}, 0, 10, false, 7);
```

---

## Accessors

Read layout and flags. None of these allocate.

| Method | Description |
| --- | --- |
| `rank()` | Number of axes (`shape_.size()`). Same as `ndim()`. |
| `ndim()` | Alias of `rank()`. |
| `numel()` | Product of `shape_`. Independent of strides, offset, and buffer size. |
| `shape()` | Logical sizes. Do not mutate the returned vector. |
| `strides()` | Per-axis step in the buffer, in elements. |
| `offset()` | Starting index into the buffer (`0` for owners). |
| `dtype()` | Runtime tag matching `T`. |
| `requires_grad()` | Whether autograd tracks this tensor. Always false for ints. |
| `is_view()` | `base_ptr_ != nullptr`. Views share `data_ptr_` with the base. |
| `is_contiguous()` | `strides_ == compute_contiguous_strides(shape_)`. Does **not** require `offset_ == 0`. |
| `grad_fn()` | Backward node, or empty on a leaf. |
| `data()` | The **entire** backing `vector<T>`, not the logical slice. For a view, `data().size()` may be `> numel()`. |
| `operator[]` | Stride-aware chained indexing. Uses `offset_ + idx * strides_[0]` and returns a `TensorAccessor`. |

Dense kernels that loop `data()[i]` for `i in 0..numel()` assume contiguous layout **and** `offset() == 0`. Use `operator[]` to inspect views.

```cpp
tensor::Tensor<float> x({2, 3}, {1,2,3,4,5,6}, false);
x.rank();           // 2
x.numel();          // 6
x.strides();        // {3, 1}
x.is_view();        // false
x.is_contiguous();  // true

float v = static_cast<float>(x[1][2]);  // 6
x[0][0] = 9.f;

auto t = x.transpose();
t.is_view();        // true
t.is_contiguous();  // false
t.strides();        // {1, 3}
t.data().data() == x.data().data();  // same buffer
static_cast<float>(t[0][1]);         // 4  (logical, via strides)
```

---

## Casting

Always a **new packed buffer** (non-contiguous sources are packed first). Float→float keeps `requires_grad` and attaches `CastBackward`; backward recasts the gradient to the input dtype. Integer results never require grad.

| Method | Description |
| --- | --- |
| `to<U>()` | Cast every element to `U`. |
| `float32()` | `to<float>()`. |
| `float64()` | `to<double>()`. |
| `int32()` | `to<int32_t>()`. Truncates toward zero. Never requires grad. |
| `int64()` | `to<int64_t>()`. Truncates toward zero. Never requires grad. |

```cpp
tensor::Tensor<double> x({2}, {1.9, -2.7}, true);
auto f = x.float32();   // Tensor<float>, still requires_grad
auto i = x.int32();     // {1, -2}, no grad
```

---

## Shape operations

Most of these are **zero-copy views** (`from_view`: new metadata, same `data_ptr_`). `reshape` / `view` / `flatten` require a contiguous source. Negative dims go through `normalize_dimension`.

| Method | Description |
| --- | --- |
| `transpose(dim0, dim1)` | Swaps two entries of `shape_` and of `strides_`. No data copy. |
| `transpose()` | `transpose(0, 1)` for rank-2 tensors. Throws otherwise. |
| `broadcast_to(shape)` | Expands to `shape` by setting stride `0` on broadcast axes. No replication in RAM. |
| `reshape(shape)` | Reinterprets the same buffer as `shape`. `numel` must match; source must be contiguous. |
| `view(shape)` | Same algorithm as `reshape` here; kept so the graph can name the op. |
| `flatten(start=0, end=-1)` | Merges axes `[start, end]` (inclusive) into one. Requires contiguity. |
| `squeeze()` | Drops every size-1 axis. Remaining strides are kept (works on non-contiguous tensors). |
| `squeeze(dim)` | Drops that axis if its size is 1; otherwise returns an unchanged view. |
| `unsqueeze(dim)` | Inserts a size-1 axis. Valid range `[-rank-1, rank]`. New stride is the next axis’s stride (or `1` at the end). |
| `contiguous()` | Packs logical order into a dense row-major buffer. |

**`contiguous()` internals.** If `is_contiguous() && offset() == 0`, returns `*this` (same `data_ptr_`, no extra `grad_fn`). Otherwise allocates `numel()` elements, copies with `index = offset + sum(idx[d] * strides[d])`, and attaches `ContiguousBackward` when grad is on (identity on values).

```cpp
tensor::Tensor<float> x({2, 3}, {1,2,3,4,5,6}, true);
auto t = x.transpose();                 // shape {3,2}, strides {1,3}, same buffer
t.reshape({6});                         // throws (not contiguous)
auto c = t.contiguous();                // new buffer {1,4,2,5,3,6}
c.reshape({6});                         // ok
auto b = x.unsqueeze(0).broadcast_to({4, 2, 3});  // leading stride 0
auto f = x.flatten();                   // {6}, same buffer
```

---

## Elementwise operations

Each op allocates a **new packed buffer** via `from_operation_result`. Binary ops require **identical shapes** (broadcast first with `broadcast_to`). If any input `requires_grad` and grad is enabled, a matching backward node is attached.

Kernels loop `data()[i]` densely — run them on contiguous tensors with `offset() == 0`, or call `contiguous()` first.

| Method | Description |
| --- | --- |
| `add(other)` | Element-wise `*this + other`. |
| `subtract(other)` | Element-wise `*this - other`. |
| `multiply(other)` | Element-wise `*this * other`. |
| `divide(other)` | Element-wise `*this / other`. Throws `std::runtime_error` on a zero denominator. |
| `neg()` | Element-wise negation. |
| `power(exponent)` | Element-wise `(*this) ** exponent` (scalar exponent). |
| `abs()` | Element-wise absolute value. |
| `exp()` | Element-wise exponential. |
| `log()` | Element-wise natural log (`std::log`, no extra domain check). |
| `sin()` / `cos()` / `tan()` | Element-wise trig. |
| `sinh()` / `cosh()` / `tanh()` | Element-wise hyperbolic trig. |
| `sigmoid()` | `1 / (1 + exp(-x))`. |
| `relu()` | `max(0, x)`. |
| `silu()` | `x * sigmoid(x)` (swish). |
| `gelu()` | Tanh approximation: `0.5 x (1 + tanh(sqrt(2/π) (x + 0.044715 x³)))`. |

```cpp
tensor::Tensor<float> a({2}, {1.f, 2.f}, true);
tensor::Tensor<float> b({2}, {3.f, 4.f}, true);
auto y = a.add(b).relu();     // {4, 6}, has grad_fn
auto s = a.silu();
```

---

## Reductions

Allocate a new packed result. Global reductions return shape `{}`. `sum(dim)` / `mean(dim)` drop that axis. Empty tensors throw on global `sum` / `mean` / `max` / `min`.

| Method | Description |
| --- | --- |
| `sum()` | Sum of every element → scalar `{}`. |
| `sum(dim)` | Sum along `dim`; that axis is removed. Negative `dim` allowed. |
| `mean()` | Mean of every element → scalar `{}`. |
| `mean(dim)` | Mean along `dim`; that axis is removed. |
| `max()` | Maximum element → scalar. Ties keep the first index (for backward). |
| `min()` | Minimum element → scalar. Ties keep the first index. |
| `softmax(dim)` | Softmax along `dim`. Same shape as input; each slice sums to 1. Uses max-subtraction for stability. |

```cpp
tensor::Tensor<float> x({2, 3}, {1,2,3,4,5,6}, true);
x.sum();          // 21, shape {}
x.sum(0);         // {5, 7, 9}
x.mean(1);        // {2, 5}
x.softmax(-1);    // same shape {2,3}, rows sum to 1
```

---

## Linear algebra

| Method | Description |
| --- | --- |
| `dot(other)` | Inner product of two 1-D tensors of equal length → scalar `{}`. |
| `matmul(other)` | Matrix product `[M, K] @ [K, N] → [M, N]`. **Reads through strides**, so transposed inputs are correct without `contiguous()`. |

```cpp
tensor::Tensor<float> A({2, 3}, {1,2,3,4,5,6}, true);
tensor::Tensor<float> B({3, 2}, {1,0, 0,1, 1,1}, true);
auto C = A.matmul(B);              // {2, 2}
auto D = A.transpose().matmul(A);  // {3, 3}, no contiguous() needed
```

---

## Autograd and internal factories

A **leaf** is a user-created tensor (`grad_fn_ == nullptr`). An op output stores a `Node` in `grad_fn_` and is not a leaf. Only leaves own `grad_storage_`.

| Method | Description |
| --- | --- |
| `is_leaf()` | `grad_fn_ == nullptr`. |
| `grad()` | Pointer to the accumulated leaf gradient, or `nullptr` if backward has not run (or this tensor does not require grad). Reads `grad_storage_->tensor`. |
| `accumulate_grad(g)` | Engine-internal. First call allocates `GradStorage::tensor`; later calls add in place. No-op if `grad_storage_` is null. |
| `zero_grad()` | Sets `GradStorage::tensor = nullptr` but **keeps the box**, so later aliases still share it. |
| `backward()` | Reverse-mode from this tensor. Requires `numel() == 1` and a non-null `grad_fn`. Seeds the engine with `1` and accumulates into reachable leaves. |
| `alias(source)` | Same buffer and same `grad_storage_` (`shared_ptr` copies). Strips `grad_fn_` (the alias is not an op). What nodes save for backward. |
| `from_view(source, shape, strides, offset, requires_grad, grad_fn)` | View factory used by shape ops. Shares `data_ptr_`, sets `base_ptr_`, clears `grad_storage_`. |
| `from_operation_result(shape, storage, requires_grad, grad_fn)` | Math-op factory. Moves `storage` into a new packed owner. |

`alias` vs `from_view`: both share `data_ptr_`. `alias` also shares `grad_storage_` and looks like the source (same shape/strides). `from_view` has new layout metadata and is a non-leaf (`grad_storage_ == nullptr`).

In-place `data()` / `operator[]` writes after forward and before backward are **not** version-checked; saved aliases will see the mutated values.

```cpp
tensor::Tensor<float> w({2, 3}, {1,2,3,4,5,6}, true);
auto loss = w.transpose().contiguous().sum();
loss.backward();
// w.grad() is all ones
w.zero_grad();              // w.grad() == nullptr; GradStorage box remains
```

---

## Helpers

Static utilities used by ops and init. Safe against `int64_t` overflow.

| Method | Description |
| --- | --- |
| `compute_numel(shape)` | Product of dimensions. Throws on a negative size. |
| `compute_contiguous_strides(shape)` | Row-major strides, e.g. `{2,3,4} → {12,4,1}`. |
| `normalize_dimension(dim, rank)` | Maps a possibly negative `dim` into `[0, rank)`. Throws `std::out_of_range` if still out of bounds. |

```cpp
tensor::Tensor<float>::compute_numel({2, 3, 4});                 // 24
tensor::Tensor<float>::compute_contiguous_strides({2, 3, 4});    // {12, 4, 1}
tensor::Tensor<float>::normalize_dimension(-1, 4);               // 3
```
