# Tensor internals

How `tensor::Tensor<T>` stores values, how views avoid copies, and when a kernel actually allocates. The public method list is in [methods.md](methods.md). How leaf gradients are written during `backward()` is in [../optim/mechanism.md](../optim/mechanism.md).

## What a tensor is

A tensor is a small metadata object plus a **shared** buffer:

```text
Tensor
  shape_        logical dimensions, e.g. [2, 3]
  strides_      step in the buffer along each axis, in elements
  offset_       starting index into the buffer
  data_ptr_     shared_ptr<vector<T>>     <-- the only heap allocation for values
  base_ptr_     shared_ptr<Tensor<T>>     <-- non-null iff this is a view
  dtype_        runtime tag matching T
  requires_grad_
  grad_fn_      backward node, or nullptr for a leaf
  grad_storage_ GradStorage box, or nullptr
```

There is no separate `Storage` type. The buffer is a `std::vector<T>` owned by `std::shared_ptr`. Copying a `Tensor` copies the metadata and **increments the refcount**; it does not clone the numbers.

The buffer lives until the last owner *or view* that holds `data_ptr_` is destroyed (RAII). There is no manual `delete`.

## Owner vs view

An **owning** tensor allocates the vector and has `base_ptr_ == nullptr`, `offset_ == 0`, and row-major strides.

A **view** is produced by `Tensor::from_view`. It:

1. Copy-constructs from the source (same `data_ptr_`).
2. Overwrites `shape_`, `strides_`, `offset_`.
3. Sets `base_ptr_` to a `shared_ptr` copy of the source handle.
4. Clears `grad_storage_` (op outputs are not leaves).

`is_view()` is `base_ptr_ != nullptr`. `data()` still returns the **whole** backing vector, which may be larger than `numel()`.

```text
owner  shape [2, 3]  strides [3, 1]  offset 0
   |
   |  data_ptr_  ----------------+
   v                             v
view   shape [3, 2]  strides [1, 3]  offset 0    (transpose)
                                     same vector {1,2,3,4,5,6}
```

Chaining views (transpose then unsqueeze) only adds metadata. The vector is allocated once.

## Address formula

Logical index `(i0, i1, ..., i_{r-1})` maps to

```text
index = offset + i0 * strides[0] + i1 * strides[1] + ...
value = data()[index]
```

`operator[]` uses this formula (via `TensorAccessor`). That is why `x.transpose()[0][1]` reads the correct element even though the buffer was not shuffled.

`data()` does **not** apply the formula. It is the raw `vector<T>`. For a view, `data().size()` can be `> numel()`, and `data()[i]` is storage order, not logical order.

## Contiguous layout

Row-major contiguous strides for shape `[d0, d1, ..., d_{n-1}]` are

```text
strides[n-1] = 1
strides[k]   = strides[k+1] * d[k+1]
```

Example: `[2, 3, 4] → [12, 4, 1]`.

`is_contiguous()` checks that `strides_ == compute_contiguous_strides(shape_)`. It does **not** require `offset_ == 0`. A packed slice can still sit at a non-zero offset.

Most elementwise and reduction kernels loop `data()[i]` for `i in 0 .. numel()`. That is correct only when the tensor is contiguous **and** `offset() == 0`. `matmul` is the exception: it reads `offset + i*s0 + k*s1`.

That is why `contiguous()` exists.

## When memory is allocated

| Path | Allocates a new `vector<T>`? |
| --- | --- |
| Constructors, `zeros` / `ones` / `full` / random / Xavier / Kaiming | Yes |
| `from_operation_result` (add, relu, sum, `to()`, …) | Yes — the op writes a dense result |
| `from_view` (`transpose`, `reshape`, `view`, `flatten`, `squeeze`, `unsqueeze`, `broadcast_to`) | No |
| `alias` | No |
| Copy / move of a `Tensor` handle | No |
| `contiguous()` of a packed tensor (`is_contiguous() && offset() == 0`) | No — returns `*this` |
| `contiguous()` of a strided view (e.g. after `transpose`) | Yes — gather via the stride formula |

`reshape` / `view` / `flatten` refuse non-contiguous inputs: they only rewrite shape and contiguous strides, they do not gather. Call `contiguous()` first.

## Shape ops and strides

**transpose(`dim0`, `dim1`)** swaps two entries of `shape_` and of `strides_`. For a `[2, 3]` matrix with strides `[3, 1]`, `transpose()` yields `[3, 2]` with strides `[1, 3]`. The six values stay put.

**broadcast_to** prepends or expands size-1 axes by setting **stride 0**. Advancing that axis does not move the pointer, so the same element is reused. No replication in RAM.

**reshape / view / flatten** require contiguity, then install `compute_contiguous_strides(new_shape)` at the same `offset_`.

**squeeze** drops size-1 axes and their strides (valid on non-contiguous tensors). **unsqueeze** inserts a size-1 axis; the new stride is the next axis’s stride (or `1` at the end).

## `alias` vs `from_view`

Both share `data_ptr_`. They differ for autograd:

| | `alias` | `from_view` |
| --- | --- | --- |
| Metadata | Identical to source | New shape / strides / offset |
| `grad_fn_` | Stripped (`nullptr`) | The shape op’s backward node |
| `grad_storage_` | **Copied** (same `shared_ptr`) | **Cleared** (non-leaf) |
| `requires_grad_` | Kept | Taken from the op |

The graph stores `alias(input)` on each node. Because leaves pre-allocate `GradStorage` at construction, the alias and the user tensor point at the same box. `accumulate_grad` on the alias is visible as `param.grad()`.

If `GradStorage` were allocated lazily on first backward, an alias created at forward time would still hold `nullptr` and leaf gradients would never land on the original parameter. See [../optim/mechanism.md](../optim/mechanism.md).

## `contiguous()` and autograd

Fast path: already packed → return the same handle, **no extra node**. Backward already flows through whatever `grad_fn_` the tensor had.

Slow path: allocate `numel()` elements, copy with the address formula, `from_operation_result` with `ContiguousBackward`. That node is the identity on values: shape is unchanged, only physical layout changed. The previous shape node (e.g. `TransposeBackward`) still sees a gradient with the logical shape.

```text
leaf [2, 3]  --transpose-->  view [3, 2]  --contiguous-->  packed [3, 2]
                 (swap strides)              (gather copy)
```

## GradStorage in one paragraph

Only **leaves** with `requires_grad == true` own a `GradStorage`. Op results have `grad_fn` and `grad_storage_ == nullptr`; their gradient lives in the engine map for one `backward()`. `zero_grad()` nulls `GradStorage::tensor` but keeps the box so the next forward’s aliases still share it. Integer tensors never get a box (`grad_allowed` is false).

## Kernel constraint (honest)

This design is efficient for **views**: transpose, broadcast, and squeeze are metadata. It is not yet efficient for **math on those views**. `add`, `relu`, `sum`, … index the raw buffer densely. `matmul` respects strides.

Practical rule: run dense math on packed tensors, or call `contiguous()` after a strided view. `operator[]` is always stride-correct, so it is the right way to inspect a view in tests.
