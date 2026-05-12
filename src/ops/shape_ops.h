#pragma once

#include <algorithm>
#include <cstdint>
#include <memory>
#include <stdexcept>

#include "../tensor/tensor.h"
#include "../autograd/grad_context.h"
#include "../autograd/backward_ops/backward_shape_ops.h"

namespace ops {

template <typename T>
tensor::Tensor<T> transpose(
    const tensor::Tensor<T>& x,
    int64_t dim0,
    int64_t dim1
) {
    const int64_t rank = x.rank();
    if (rank == 0)
        throw std::invalid_argument("cannot transpose a scalar tensor");

    dim0 = tensor::Tensor<T>::normalize_dimension(dim0, rank);
    dim1 = tensor::Tensor<T>::normalize_dimension(dim1, rank);

    auto shape   = x.shape();
    auto strides = x.strides();

    if (dim0 != dim1) {
        std::swap(shape  [static_cast<size_t>(dim0)], shape  [static_cast<size_t>(dim1)]);
        std::swap(strides[static_cast<size_t>(dim0)], strides[static_cast<size_t>(dim1)]);
    }

    const bool requires_grad = autograd::is_grad_enabled() && x.requires_grad();
    std::shared_ptr<autograd::Node<T>> grad_fn = nullptr;
    if (requires_grad)
        grad_fn = std::make_shared<autograd::TransposeBackward<T>>(x, dim0, dim1);

    return tensor::Tensor<T>::from_view(
        x,
        std::move(shape),
        std::move(strides),
        x.offset(),
        requires_grad,
        std::move(grad_fn)
    );
}

template <typename T>
tensor::Tensor<T> reshape(
    const tensor::Tensor<T>& x,
    const std::vector<int64_t>& shape
) {
    const int64_t rank = x.rank();
    if (rank == 0)
        throw std::invalid_argument("cannot reshape a scalar tensor");

    const int64_t numel = tensor::Tensor<T>::compute_numel(shape);
    if (numel != x.numel())
        throw std::invalid_argument("cannot reshape tensor with different number of elements");

    if (!x.is_contiguous())
        throw std::invalid_argument("cannot reshape a non-contiguous tensor; call contiguous() first");

    const bool requires_grad = autograd::is_grad_enabled() && x.requires_grad();
    std::shared_ptr<autograd::Node<T>> grad_fn = nullptr;
    if (requires_grad)
        grad_fn = std::make_shared<autograd::ReshapeBackward<T>>(x, x.shape());

    return tensor::Tensor<T>::from_view(
        x,
        shape,
        tensor::Tensor<T>::compute_contiguous_strides(shape),
        x.offset(),
        requires_grad,
        std::move(grad_fn)
    );
}

// Compute the shape that results from broadcasting `a` and `b` together.
//
// Follows the NumPy/PyTorch right-alignment rule:
//   - The shorter shape is implicitly left-padded with 1s.
//   - For each aligned pair (da, db):
//       da == db  → output dim is da
//       da == 1   → output dim is db  (a is broadcast along this axis)
//       db == 1   → output dim is da  (b is broadcast along this axis)
//       otherwise → shapes are incompatible, throw
inline std::vector<int64_t> broadcast_shapes(
    const std::vector<int64_t>& a,
    const std::vector<int64_t>& b
) {
    const size_t rank = std::max(a.size(), b.size());
    std::vector<int64_t> out(rank);

    for (size_t i = 0; i < rank; ++i) {
        // Align from the right: pad the shorter shape with virtual 1s on the left.
        const int64_t da = (i < rank - a.size()) ? 1 : a[i - (rank - a.size())];
        const int64_t db = (i < rank - b.size()) ? 1 : b[i - (rank - b.size())];

        if      (da == db) out[i] = da;
        else if (da == 1)  out[i] = db;
        else if (db == 1)  out[i] = da;
        else
            throw std::invalid_argument(
                "shapes are not broadcastable: dimension " + std::to_string(i) +
                " has sizes " + std::to_string(da) + " and " + std::to_string(db));
    }
    return out;
}

// Return a zero-copy view of `x` with shape `target_shape`.
//
// No memory is allocated and no data is copied.  Broadcasting is achieved via
// the stride-0 trick: any axis where x.shape[i] == 1 and target[i] > 1 gets its
// stride set to 0, so iterating along that axis always reads the same element.
//
// Compatibility rules (NumPy / PyTorch):
//   - target_rank >= x.rank() is required.
//   - Extra leading dims in target are treated as if x had size 1 there (stride 0).
//   - Aligned dims must satisfy: x.shape[i] == target[i]  (keep stride)
//                             or: x.shape[i] == 1          (set stride to 0)
//
// Backward: dL/dx = sum(dL/d_out) over all broadcast axes, because each original
// element contributed to N output positions and all those gradients must be
// summed back (see BroadcastToBackward / sum_to).
template <typename T>
tensor::Tensor<T> broadcast_to(
    const tensor::Tensor<T>& x,
    const std::vector<int64_t>& target_shape
) {
    const int64_t x_rank = x.rank();
    const int64_t target_rank = static_cast<int64_t>(target_shape.size());

    if (target_rank < x_rank)
        throw std::invalid_argument(
            "broadcast_to: target rank (" + std::to_string(target_rank) +
            ") must be >= source rank (" + std::to_string(x_rank) + ")");

    // Number of leading dimensions that need to be prepended (virtual size 1).
    const int64_t rank_diff = target_rank - x_rank;

    std::vector<int64_t> new_strides(static_cast<size_t>(target_rank));

    for (int64_t d = 0; d < target_rank; ++d) {
        const int64_t target_dim = target_shape[static_cast<size_t>(d)];

        if (d < rank_diff) {
            // Prepended dim: x has no corresponding axis, behaves as size 1.
            // Stride 0 means advancing along this axis never moves the pointer.
            new_strides[static_cast<size_t>(d)] = 0;
        } else {
            const int64_t src = static_cast<size_t>(d - rank_diff);
            const int64_t x_dim    = x.shape()  [static_cast<size_t>(src)];
            const int64_t x_stride = x.strides()[static_cast<size_t>(src)];

            if (x_dim == target_dim) {
                // Dimensions match: keep the original stride unchanged.
                new_strides[static_cast<size_t>(d)] = x_stride;
            } else if (x_dim == 1) {
                // Size-1 dim: broadcast by collapsing the stride to 0.
                new_strides[static_cast<size_t>(d)] = 0;
            } else {
                throw std::invalid_argument(
                    "broadcast_to: dim " + std::to_string(src) +
                    " has size " + std::to_string(x_dim) +
                    " which cannot broadcast to " + std::to_string(target_dim));
            }
        }
    }

    const bool requires_grad = autograd::is_grad_enabled() && x.requires_grad();
    std::shared_ptr<autograd::Node<T>> grad_fn = nullptr;
    if (requires_grad)
        grad_fn = std::make_shared<autograd::BroadcastToBackward<T>>(x, x.shape());

    return tensor::Tensor<T>::from_view(
        x,
        target_shape,
        std::move(new_strides),
        x.offset(),
        requires_grad,
        std::move(grad_fn)
    );
}

// view: shares storage with source, identical semantics to reshape in this
// codebase (requires contiguous, never copies). Kept as a distinct op so the
// autograd graph can distinguish the operation.
template <typename T>
tensor::Tensor<T> view(
    const tensor::Tensor<T>& x,
    const std::vector<int64_t>& shape
) {
    return reshape<T>(x, shape);
}

// flatten: collapses the range [start_dim, end_dim] (inclusive) into a single
// dimension. Requires the tensor to be contiguous.
//
// Example: shape [2,3,4], flatten(1,-1) → [2,12]
template <typename T>
tensor::Tensor<T> flatten(
    const tensor::Tensor<T>& x,
    int64_t start_dim,
    int64_t end_dim
) {
    const int64_t rank = x.rank();
    if (rank == 0)
        throw std::invalid_argument("cannot flatten a scalar tensor");

    start_dim = tensor::Tensor<T>::normalize_dimension(start_dim, rank);
    end_dim   = tensor::Tensor<T>::normalize_dimension(end_dim,   rank);

    if (start_dim > end_dim)
        throw std::invalid_argument("flatten: start_dim must be <= end_dim after normalization");

    if (!x.is_contiguous())
        throw std::invalid_argument("cannot flatten a non-contiguous tensor; call contiguous() first");

    std::vector<int64_t> new_shape;
    new_shape.reserve(static_cast<size_t>(rank - (end_dim - start_dim)));

    for (int64_t i = 0; i < start_dim; ++i)
        new_shape.push_back(x.shape()[static_cast<size_t>(i)]);

    int64_t flat_size = 1;
    for (int64_t i = start_dim; i <= end_dim; ++i)
        flat_size *= x.shape()[static_cast<size_t>(i)];
    new_shape.push_back(flat_size);

    for (int64_t i = end_dim + 1; i < rank; ++i)
        new_shape.push_back(x.shape()[static_cast<size_t>(i)]);

    const bool requires_grad = autograd::is_grad_enabled() && x.requires_grad();
    std::shared_ptr<autograd::Node<T>> grad_fn = nullptr;
    if (requires_grad)
        grad_fn = std::make_shared<autograd::FlattenBackward<T>>(x, x.shape());

    return tensor::Tensor<T>::from_view(
        x,
        new_shape,
        tensor::Tensor<T>::compute_contiguous_strides(new_shape),
        x.offset(),
        requires_grad,
        std::move(grad_fn)
    );
}

// squeeze(): remove all size-1 dimensions. The stride associated with each
// removed dimension is dropped; remaining strides are preserved as-is so the
// op is valid on non-contiguous tensors.
template <typename T>
tensor::Tensor<T> squeeze(const tensor::Tensor<T>& x) {
    std::vector<int64_t> new_shape;
    std::vector<int64_t> new_strides;

    const auto& shape   = x.shape();
    const auto& strides = x.strides();
    for (size_t i = 0; i < shape.size(); ++i) {
        if (shape[i] != 1) {
            new_shape.push_back(shape[i]);
            new_strides.push_back(strides[i]);
        }
    }

    const bool requires_grad = autograd::is_grad_enabled() && x.requires_grad();
    std::shared_ptr<autograd::Node<T>> grad_fn = nullptr;
    if (requires_grad)
        grad_fn = std::make_shared<autograd::SqueezeBackward<T>>(x, x.shape());

    return tensor::Tensor<T>::from_view(
        x,
        std::move(new_shape),
        std::move(new_strides),
        x.offset(),
        requires_grad,
        std::move(grad_fn)
    );
}

// squeeze(dim): remove the specified dimension only if its size is 1.
// If the dimension does not have size 1 the tensor is returned unchanged
// (following PyTorch convention).
template <typename T>
tensor::Tensor<T> squeeze(const tensor::Tensor<T>& x, int64_t dim) {
    const int64_t rank = x.rank();
    dim = tensor::Tensor<T>::normalize_dimension(dim, rank);

    if (x.shape()[static_cast<size_t>(dim)] != 1) {
        // Dimension is not size 1; return unchanged view.
        return tensor::Tensor<T>::from_view(
            x, x.shape(), x.strides(), x.offset(), x.requires_grad(), nullptr
        );
    }

    std::vector<int64_t> new_shape   = x.shape();
    std::vector<int64_t> new_strides = x.strides();
    new_shape.erase(  new_shape.begin()   + dim);
    new_strides.erase(new_strides.begin() + dim);

    const bool requires_grad = autograd::is_grad_enabled() && x.requires_grad();
    std::shared_ptr<autograd::Node<T>> grad_fn = nullptr;
    if (requires_grad)
        grad_fn = std::make_shared<autograd::SqueezeBackward<T>>(x, x.shape());

    return tensor::Tensor<T>::from_view(
        x,
        std::move(new_shape),
        std::move(new_strides),
        x.offset(),
        requires_grad,
        std::move(grad_fn)
    );
}

// unsqueeze(dim): insert a new size-1 dimension at position dim.
// dim may be negative; valid range is [-rank-1, rank] (same as PyTorch).
// The inserted stride matches the stride of the immediately following dimension
// (or 1 if inserted at the end), matching PyTorch's contiguous-stride convention.
template <typename T>
tensor::Tensor<T> unsqueeze(const tensor::Tensor<T>& x, int64_t dim) {
    const int64_t rank = x.rank();
    // Valid range is [-(rank+1), rank]; normalize to [0, rank].
    if (dim < 0)
        dim += rank + 1;
    if (dim < 0 || dim > rank)
        throw std::out_of_range("unsqueeze: dimension index out of range");

    std::vector<int64_t> new_shape   = x.shape();
    std::vector<int64_t> new_strides = x.strides();

    new_shape.insert(  new_shape.begin()   + dim, 1);
    // Stride for the new size-1 dim: use the stride of the next existing dim,
    // or 1 if inserted at the end.
    const int64_t inserted_stride = (dim < rank) ? x.strides()[static_cast<size_t>(dim)] : 1;
    new_strides.insert(new_strides.begin() + dim, inserted_stride);

    const bool requires_grad = autograd::is_grad_enabled() && x.requires_grad();
    std::shared_ptr<autograd::Node<T>> grad_fn = nullptr;
    if (requires_grad)
        grad_fn = std::make_shared<autograd::UnsqueezeBackward<T>>(x, x.shape());

    return tensor::Tensor<T>::from_view(
        x,
        std::move(new_shape),
        std::move(new_strides),
        x.offset(),
        requires_grad,
        std::move(grad_fn)
    );
}

} // namespace ops
