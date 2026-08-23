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

A tensor is **contiguous** when `strides[i] == product of shape[i+1 .. rank-1]` for every dimension (standard row-major layout). Several ops require contiguity — see [Contiguity requirements](#contiguity-requirements).

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

The `data` vector must have exactly `numel(shape)` elements; otherwise `std::invalid_argument` is thrown.

Integer tensors silently ignore `requires_grad = true` — gradients are never tracked for integer element types.

### Factory constructors (static)

All factories accept an optional `requires_grad` flag (default `false`).

| Factory | Description |
|---|---|
| `Tensor<T>::zeros(shape)` | All elements `0` |
| `Tensor<T>::ones(shape)` | All elements `1` |
| `Tensor<T>::full(shape, value)` | All elements `value` |
| `Tensor<T>::zeros_like(other)` | Same shape as `other`, all zeros |
| `Tensor<T>::ones_like(other)` | Same shape as `other`, all ones |
| `Tensor<T>::full_like(other, value)` | Same shape as `other`, filled with `value` |
| `Tensor<T>::randint(shape, low, high)` | Uniform integer random in `[low, high)` |
| `Tensor<T>::randn(shape)` | Standard normal (mean 0, stddev 1) |
| `Tensor<T>::random_gaussian(shape, mean, stddev)` | Gaussian with given parameters |
| `Tensor<T>::randint_like(other, low, high)` | Random integers, same shape as `other` |
| `Tensor<T>::randn_like(other)` | Standard normal, same shape as `other` |
| `Tensor<T>::random_gaussian_like(other, mean, stddev)` | Gaussian, same shape as `other` |

All random factories accept an optional `seed` parameter for reproducibility:

```cpp
auto r = Tensor<float>::randn({4, 4}, false, /*seed=*/42);
```

## Accessors

```cpp
int64_t              rank()          const;   // number of dimensions
int64_t              ndim()          const;   // alias for rank()
int64_t              numel()         const;   // total number of elements
std::vector<int64_t> shape()         const;
std::vector<int64_t> strides()       const;
int64_t              offset()        const;   // offset into the data buffer
Dtype                dtype()         const;
bool                 requires_grad() const;
bool                 is_view()       const;
bool                 is_contiguous() const;

// autograd
std::shared_ptr<autograd::Node<T>> grad_fn()  const;
bool                               is_leaf()  const;  // grad_fn == nullptr

// raw data access
std::vector<T>&       data();
const std::vector<T>& data() const;
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

Creates a new owning tensor with elements cast via `static_cast<U>`. The result always has `requires_grad = false` and `grad_fn = nullptr` — casting detaches from the computation graph.

```cpp
Tensor<float>   f({3}, {1.f, 2.f, 3.f});
Tensor<double>  d = f.to<double>();
Tensor<int32_t> i = f.to<int32_t>();
```

## Shape operations

All shape ops are available as free functions in `ops::` and as methods on `Tensor`. See [ops.md](ops.md#shape-operations) for full details.

```cpp
Tensor<float> t({2, 3, 4}, data);

auto r  = t.reshape({6, 4});           // must be contiguous
auto tr = t.transpose(0, 2);          // swap dims 0 and 2
auto bc = t.broadcast_to({5, 2, 3, 4});
auto v  = t.view({24});               // alias for reshape
auto fl = t.flatten(1, 2);            // flatten dims 1..2 → {2, 12}
auto sq = t.squeeze();                // remove all size-1 dims
auto us = t.unsqueeze(1);             // insert size-1 dim at position 1
```

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
auto z = x.sin();
auto z = x.cos();
auto z = x.tan();
auto z = x.sigmoid();
auto z = x.relu();
```

All binary elementwise ops require **matching shapes**. There is no implicit broadcasting — use `broadcast_to` first.

## Reduction operations

```cpp
auto s  = x.sum();          // global sum → scalar
auto s  = x.sum(dim);       // reduce along dim → rank-1 fewer tensor
auto m  = x.mean();         // global mean → scalar
auto m  = x.mean(dim);      // reduce along dim
auto mx = x.max();          // global maximum → scalar
auto mn = x.min();          // global minimum → scalar
```

## Linear algebra

```cpp
auto d = ops::dot(a, b);      // 1-D × 1-D → scalar
auto c = ops::matmul(a, b);   // 2-D × 2-D → 2-D
```

`dot` requires rank-1 inputs of the same length. `matmul` requires rank-2 with compatible inner dimensions. Both are stride-safe.

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
std::shared_ptr<Tensor<T>> g = x.grad();
```

### Running the backward pass

```cpp
loss.backward();   // loss must be a scalar (numel() == 1)
```

`backward()` throws `std::invalid_argument` if `numel() != 1` or if `grad_fn_` is null.

The backward pass uses a topological sort of the computation graph and accumulates gradients via the chain rule. See [autograd.md](autograd.md) for details on the engine.

### Clearing gradients

```cpp
x.zero_grad();   // sets the AccumulateGrad buffer to zero
```

Call this before each training iteration to prevent gradient accumulation across steps.

## Contiguity requirements

The following operations require the input tensor to be contiguous (`is_contiguous() == true`):

| Operation | Reason |
|---|---|
| `reshape` | Must reinterpret the flat buffer |
| `view` | Same as `reshape` |
| `flatten` | Collapses a dimension range |
| `elementwise ops` | Read `data()[i]` sequentially |
| `sum(dim)` / `mean(dim)` | Use contiguous stride formula for indexing |

Operations that are **stride-safe** (work on non-contiguous tensors):

| Operation | Notes |
|---|---|
| `transpose` | Returns a view with swapped strides |
| `broadcast_to` | Sets stride-0 on expanded dims |
| `squeeze` / `unsqueeze` | Metadata-only changes |
| `matmul` | Full stride/offset formula |
| `dot` | Full stride formula |

If you need to make a view contiguous before passing it to a contiguous-only op, materialise a new owning tensor:

```cpp
// manual contiguous copy (no built-in .contiguous() method yet)
auto flat = Tensor<float>(t.shape(), t.data(), t.requires_grad());
```

## Internal factory helpers (advanced)

These static methods are used by `ops::` implementations and are not intended for general use:

```cpp
// output of a forward op: owns new storage, optional grad_fn
static Tensor<T> from_operation_result(
    shape, storage&&, requires_grad, grad_fn, dtype);

// view: shares storage with source, own metadata
static Tensor<T> from_view(
    source, shape, strides, offset, requires_grad, grad_fn);
```
