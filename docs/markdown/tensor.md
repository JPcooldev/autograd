# Tensor

`tensor::Tensor<T>` is the central data structure of the library. It represents an n-dimensional array of elements of type `T`, together with optional gradient tracking metadata.

Supported element types:

| C++ type | `Dtype` enum | Grad support |
|---|---|---|
| `float` | `Dtype::Float32` | yes |
| `double` | `Dtype::Float64` | yes |
| `int32_t` | `Dtype::Int32` | no |
| `int64_t` | `Dtype::Int64` | no |

## Memory model

Every `Tensor` is either an **owning tensor** or a **view**.

- An **owning tensor** holds a `shared_ptr<vector<T>>` to the actual data buffer it allocated.
- A **view** holds the same `shared_ptr` as the source tensor (reference-counted, zero copy) and carries its own `shape`, `strides`, and `offset`.

Whether a tensor is a view is indicated by `is_view()`. Multiple levels of view chaining are supported; the buffer stays alive as long as any tensor in the chain is alive.

### Stride formula

Every element access uses:

```
flat_index = offset + sum( logical_index[i] * strides[i] )
```

A tensor is **contiguous** when `strides[i] == product of shape[i+1 .. rank-1]` for every dimension (standard row-major layout). Dense math packs at the kernel door; see [Contiguity requirements](#contiguity-requirements).

## Dtype

```cpp
// header: src/tensor/dtype.h

enum class Dtype { Float32, Float64, Int32, Int64 };

Dtype get_default_dtype();           // Float32 by default (thread-local)
void  set_default_dtype(Dtype d);
void  reset_default_dtype();         // back to Float32
```

The default dtype is thread-local, so it can be changed safely per thread.

## Constructors

### Direct construction

```cpp
// shape + data vector
Tensor<float> a({2, 3}, {1.f, 2.f, 3.f, 4.f, 5.f, 6.f});

// shape + data + requires_grad
Tensor<float> x({3}, {1.f, 2.f, 3.f}, /*requires_grad=*/true);

// scalar (rank-0) tensor
Tensor<float> s({}, {3.14f});
```

The `data` vector must have exactly `numel(shape)` elements; otherwise `std::invalid_argument` is thrown. The two- and three-argument constructors default `requires_grad` to **true** (forced off for integer `T`).

Integer tensors silently ignore `requires_grad = true` — gradients are never tracked for integer element types.

### Factory constructors (static)

Fill / random factories accept an optional `requires_grad` flag (**default `true`**, forced off for integer `T`). `randint` is the exception: it never takes `requires_grad` and is always `false`.

| Factory | Description |
|---|---|
| `Tensor<T>::zeros(shape)` | All elements `0` |
| `Tensor<T>::ones(shape)` | All elements `1` |
| `Tensor<T>::full(shape, value)` | All elements `value` |
| `Tensor<T>::zeros_like(other)` | Same shape as `other`, all zeros |
| `Tensor<T>::ones_like(other)` | Same shape as `other`, all ones |
| `Tensor<T>::full_like(other, value)` | Same shape as `other`, filled with `value` |
| `Tensor<T>::arange(start, end, step=1)` | 1-D `[start, end)` |
| `Tensor<T>::randint(shape, low, high)` | Uniform integer random in `[low, high)`; no grad |
| `Tensor<T>::randn(shape)` | Standard normal (mean 0, stddev 1) |
| `Tensor<T>::random_gaussian(shape, mean, stddev)` | Gaussian with given parameters |
| `Tensor<T>::uniform(shape, low, high)` | Uniform `U[low, high)` |
| `Tensor<T>::randint_like(other, low, high)` | Random integers, same shape as `other` |
| `Tensor<T>::randn_like(other)` | Standard normal, same shape as `other` |
| `Tensor<T>::random_gaussian_like(other, mean, stddev)` | Gaussian, same shape as `other` |
| `Tensor<T>::xavier_uniform` / `xavier_normal` | Glorot init |
| `Tensor<T>::kaiming_uniform` / `kaiming_normal` | He init |

Random factories accept an optional `seed` (`std::optional<uint64_t>`) after `requires_grad` (after `high` for `randint`):

```cpp
auto r = Tensor<float>::randn({4, 4}, false, /*seed=*/42);
auto g = Tensor<float>::random_gaussian({4}, 0.0f, 1.0f, true, /*seed=*/42);
auto a = Tensor<float>::arange(0, 5);   // {0,1,2,3,4}
```

The complete signature list is in [tensor/methods.md](tensor/methods.md).

## Accessors

```cpp
int64_t                    rank()          const;
int64_t                    ndim()          const;
int64_t                    numel()         const;
const std::vector<int64_t>& shape()        const;
const std::vector<int64_t>& strides()      const;
int64_t                    offset()        const;
Dtype                      dtype()         const;
bool                       requires_grad() const;
bool                       is_view()       const;
bool                       is_contiguous() const;

std::shared_ptr<autograd::Node<T>> grad_fn() const;
bool                               is_leaf() const;  // grad_fn == nullptr
const Tensor<T>*                   grad()    const;

std::vector<T>&       data();
const std::vector<T>& data() const;
std::vector<T>        to_vector() const;
void                  copy_(const Tensor& src);
```

### Element access via `operator[]`

Chained subscript returns a `TensorAccessor` that enforces bounds checking:

```cpp
Tensor<float> t({2, 3}, {1, 2, 3, 4, 5, 6});
float v = t[1][2];   // 6.0f

t[0][1] = 99.f;      // mutable access
```

## Casting

```cpp
template <typename U>
Tensor<U> to() const;
```

Creates a new packed tensor with elements cast via `static_cast<U>`. Float→float keeps `requires_grad` and attaches `CastBackward`. Integer results never require grad (`float32()` / `float64()` / `int32()` / `int64()` are shorthands).

```cpp
Tensor<float>   f({3}, {1.f, 2.f, 3.f}, true);
Tensor<double>  d = f.to<double>();     // still requires_grad
Tensor<int32_t> i = f.to<int32_t>();    // no grad
```

## Shape operations

All shape ops are available as free functions in `ops::` and as methods on `Tensor`. See [ops.md](ops.md#shape-operations) for full details.

```cpp
Tensor<float> t({2, 3, 4}, data);

auto r  = t.reshape({6, 4});           // packs if needed
auto tr = t.transpose(0, 2);          // swap dims 0 and 2
auto bc = t.broadcast_to({5, 2, 3, 4});
auto v  = t.view({24});               // zero-copy; source must be contiguous
auto fl = t.flatten(1, 2);            // flatten dims 1..2 → {2, 12}
auto sq = t.squeeze();                // remove all size-1 dims
auto us = t.unsqueeze(1);             // insert size-1 dim at position 1
auto nw = t.narrow(0, 0, 1);          // view of the first row
auto packed = t.transpose().contiguous();
```

`ops::cat({a, b}, dim)` concatenates along an axis (not a Tensor method).

The 0-argument `transpose()` is only valid for rank-2 tensors and swaps the two dimensions.

## Elementwise operations

Available as `ops::` free functions and `Tensor` methods:

```cpp
auto z = x.add(y);
auto z = x.subtract(y);
auto z = x.multiply(y);
auto z = x.divide(y);
auto z = x.power(2.0f);      // scalar exponent
auto z = x.neg();
auto z = x.abs();
auto z = x.exp();
auto z = x.log();
auto z = x.sqrt();
auto z = x.sin();
auto z = x.cos();
auto z = x.tan();
auto z = x.sinh();
auto z = x.tanh();
auto z = x.sigmoid();
auto z = x.relu();
auto z = x.silu();
auto z = x.gelu();
```

All binary elementwise ops require **matching shapes**. There is no implicit broadcasting — use `broadcast_to` first.

## Reduction operations

```cpp
auto s  = x.sum();          // global sum → scalar
auto s  = x.sum(dim);       // reduce along dim → rank-1 fewer tensor
auto m  = x.mean();         // global mean → scalar
auto m  = x.mean(dim);      // reduce along dim
auto mx = x.max();          // global maximum → scalar
auto mx = x.max(dim);       // reduce along dim
auto mn = x.min();          // global minimum → scalar
auto mn = x.min(dim);       // reduce along dim
auto sm = x.softmax(dim);   // same shape; slices along dim sum to 1
```

## Linear algebra

```cpp
auto d = ops::dot(a, b);      // 1-D × 1-D → scalar
auto c = ops::matmul(a, b);   // (..., M, K) × (..., K, N) → (..., M, N)
```

`dot` requires rank-1 inputs of the same length (packed at the kernel door). `matmul` multiplies the last two dimensions (rank ≥ 2) and broadcasts leading batch dims; it is stride-safe.

## Autograd interface

### Gradient tracking

```cpp
bool rg = x.requires_grad();   // is grad tracking enabled for this tensor?
bool il = x.is_leaf();         // true when grad_fn == nullptr (user-created tensors)
```

### Accessing gradients

```cpp
// Returns a pointer to the accumulated gradient tensor.
// nullptr if no gradient has been computed yet.
const Tensor<T>* g = x.grad();
```

### Running the backward pass

```cpp
loss.backward();   // loss must be a scalar (numel() == 1)
```

`backward()` throws `std::invalid_argument` if `numel() != 1` or if `grad_fn_` is null.

The backward pass uses a topological sort of the computation graph and accumulates gradients via the chain rule. See [autograd.md](autograd.md) for details on the engine.

### Clearing gradients

```cpp
x.zero_grad();   // drops the stored gradient; keeps the GradStorage box
```

Call this before each training iteration to prevent gradient accumulation across steps.

## Contiguity requirements

Dense kernels (`add`, `relu`, `sum`, `dot`, …) call `contiguous()` at the start, so they are correct on views. Shape ops stay views. `view` is the exception that still requires a contiguous source.

Why this, and not a strided `data()[i]` wrapper or a gather in every kernel, is in [tensor/mechanism.md](tensor/mechanism.md#why-dense-kernels-pack-at-the-door). Short version: keep `data()` as the raw buffer, keep kernels as a simple loop the compiler can vectorize, pack only when you actually run dense math (not after every `transpose` / `broadcast_to`).

| Operation | Notes |
|---|---|
| `view` | Zero-copy only; throws if not contiguous |
| `reshape` / `flatten` | Pack if a view is impossible, then rewrite shape |
| `elementwise` / `sum` / `mean` / `dot` / loss | Pack at the kernel door; `data()[i]` is then logical order |
| `matmul` | Stride-safe; does not pack |

Operations that are **stride-safe** (work on non-contiguous tensors without packing):

| Operation | Notes |
|---|---|
| `transpose` | Returns a view with swapped strides |
| `broadcast_to` | Sets stride-0 on expanded dims |
| `squeeze` / `unsqueeze` / `narrow` | Metadata-only changes |
| `matmul` | Full stride/offset formula |

If you need a packed buffer without running a math kernel, call `contiguous()`:

```cpp
auto packed = t.contiguous();  // no-op if already packed with offset == 0
```

## Internal factory helpers (advanced)

These static methods are used by `ops::` implementations and are not intended for general use:

```cpp
// output of a forward op: owns new storage, optional grad_fn
static Tensor<T> from_operation_result(
    shape, storage&&, requires_grad, grad_fn);

// view: shares storage with source, own metadata
static Tensor<T> from_view(
    source, shape, strides, offset, requires_grad, grad_fn);
```
