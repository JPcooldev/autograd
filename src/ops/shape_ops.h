#pragma once

#include <algorithm>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <stdexcept>
#include <vector>

#include "../tensor/tensor.h"
#include "../autograd/grad_context.h"
#include "../autograd/backward_ops/backward_shape_ops.h"
#include "../logging/logger.h"

namespace ops {

/**
 * Return a tensor whose logical order is packed into a dense row-major buffer.
 * If `x` is already contiguous with `offset() == 0`, returns `x`; otherwise copies
 * via `copy_to_contiguous_storage` and may attach `ContiguousBackward`.
 *
 * @param x The tensor to pack.
 * @return A contiguous tensor with the same shape and logical values.
 */
template <typename T>
tensor::Tensor<T> contiguous(const tensor::Tensor<T>& x);

/**
 * Transpose the tensor. Transpose is a view operation that only swaps the dimensions metadata (shape and strides) 
 * but keeps the data storage and offset the same.
 *
 * @param x The tensor to transpose.
 * @param dim0 The first dimension to transpose.
 * @param dim1 The second dimension to transpose.
 * @return The transposed tensor.
 *
 * @throws std::invalid_argument if the tensor is a scalar.
 * @throws std::out_of_range if the dimension indices are out of range.
 */
template <typename T>
tensor::Tensor<T> transpose(
    const tensor::Tensor<T>& x,
    int64_t dim0,
    int64_t dim1) {
    const int64_t rank = x.rank();
    if (rank == 0)
        throw std::invalid_argument("cannot transpose a scalar tensor");

    // dimensions for transpose
    dim0 = tensor::Tensor<T>::normalize_dimension(dim0, rank);
    dim1 = tensor::Tensor<T>::normalize_dimension(dim1, rank);

    // swap dimensions metadata (shape and strides)
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
        x, std::move(shape), std::move(strides), x.offset(), requires_grad, std::move(grad_fn)
    );
}

/**
 * Reshape the tensor. Reshape is a view operation that changes the dimensions metadata (shape) 
 * but keeps the data storage and offset the same.
 * If the tensor is not contiguous, it will be packed into a new data storage.
 *
 * @param x The tensor to reshape.
 * @param shape The new shape.
 * @return The reshaped tensor.
 *
 * @throws std::invalid_argument if the tensor is a scalar.
 * @throws std::invalid_argument if the new shape has a different number of elements.
 */
template <typename T>
tensor::Tensor<T> reshape(
    const tensor::Tensor<T>& x,
    const std::vector<int64_t>& shape) {
    if (x.rank() == 0)
        throw std::invalid_argument("cannot reshape a scalar tensor");

    const int64_t numel = tensor::Tensor<T>::compute_numel(shape);
    if (numel != x.numel())
        throw std::invalid_argument("cannot reshape tensor with different number of elements");

    // Metadata-only when already contiguous (including a packed slice at a
    // non-zero offset). Otherwise pack logical order, then view.
    const tensor::Tensor<T> src = x.is_contiguous() ? x : x.contiguous();

    const bool requires_grad = autograd::is_grad_enabled() && src.requires_grad();
    std::shared_ptr<autograd::Node<T>> grad_fn = nullptr;
    if (requires_grad)
        grad_fn = std::make_shared<autograd::ReshapeBackward<T>>(src, src.shape());
    auto strides = tensor::Tensor<T>::compute_contiguous_strides(shape);
    return tensor::Tensor<T>::from_view(
        src, shape, std::move(strides), src.offset(), requires_grad, std::move(grad_fn)
    );
}

/**
 * View the tensor. View is a view operation that changes the dimensions metadata
 * (shape) but keeps the data storage and offset the same. Unlike reshape, it
 * never packs — it throws if the source is not contiguous. Kept as a distinct
 * op so the autograd graph can name it.
 *
 * @param x The tensor to view.
 * @param shape The new shape.
 * @return The viewed tensor.
 *
 * @throws std::invalid_argument if the tensor is not contiguous.
 * @throws std::invalid_argument if the tensor is a scalar.
 * @throws std::invalid_argument if the new shape has a different number of elements.
 */
template <typename T>
tensor::Tensor<T> view(const tensor::Tensor<T>& x, const std::vector<int64_t>& shape) {
    if (!x.is_contiguous())
        throw std::invalid_argument("cannot view a non-contiguous tensor; call contiguous() first");
    return reshape<T>(x, shape);
}

/**
 * Compute the shape that results from broadcasting `a` and `b` together.
 * Follows the NumPy/PyTorch right-alignment rule: the shorter shape is
 * left-padded with 1s; matching sizes or a size of 1 are compatible.
 *
 * @param a First shape.
 * @param b Second shape.
 * @return The broadcast output shape.
 *
 * @throws std::invalid_argument if a pair of aligned dimensions is incompatible.
 */
inline std::vector<int64_t> broadcast_shapes(
    const std::vector<int64_t>& a,
    const std::vector<int64_t>& b) {
    const size_t rank = std::max(a.size(), b.size());
    std::vector<int64_t> out(rank);

    for (size_t i = 0; i < rank; ++i) {
        const int64_t da = (i < rank - a.size()) ? 1 : a[i - (rank - a.size())];
        const int64_t db = (i < rank - b.size()) ? 1 : b[i - (rank - b.size())];

        if (da == db)
            out[i] = da;
        else if (da == 1)
            out[i] = db;
        else if (db == 1)
            out[i] = da;
        else
            throw std::invalid_argument(
                "shapes are not broadcastable: dimension " + std::to_string(i) +
                " has sizes " + std::to_string(da) + " and " + std::to_string(db));
    }
    return out;
}

/**
 * Broadcast a tensor to `target_shape`. Returns a zero-copy view using the
 * stride-0 trick: size-1 axes that expand, and extra leading dims, get stride 0.
 *
 * @param x The tensor to broadcast.
 * @param target_shape Compatible target shape (rank at least `x.rank()`).
 * @return A view with shape `target_shape`.
 *
 * @throws std::invalid_argument if `target_shape` has smaller rank than `x`.
 * @throws std::invalid_argument if a source dimension cannot broadcast to the
 *         corresponding target size.
 */
template <typename T>
tensor::Tensor<T> broadcast_to(
    const tensor::Tensor<T>& x,
    const std::vector<int64_t>& target_shape) {
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

        if (d < rank_diff)
            new_strides[static_cast<size_t>(d)] = 0;
        else {
            const int64_t src = static_cast<size_t>(d - rank_diff);
            const int64_t x_dim    = x.shape()  [static_cast<size_t>(src)];
            const int64_t x_stride = x.strides()[static_cast<size_t>(src)];

            if (x_dim == target_dim)
                new_strides[static_cast<size_t>(d)] = x_stride;
            else if (x_dim == 1)
                new_strides[static_cast<size_t>(d)] = 0;
            else
                throw std::invalid_argument(
                    "broadcast_to: dim " + std::to_string(src) +
                    " has size " + std::to_string(x_dim) +
                    " which cannot broadcast to " + std::to_string(target_dim));
        }
    }

    const bool requires_grad = autograd::is_grad_enabled() && x.requires_grad();
    std::shared_ptr<autograd::Node<T>> grad_fn = nullptr;
    if (requires_grad)
        grad_fn = std::make_shared<autograd::BroadcastToBackward<T>>(x, x.shape());
    return tensor::Tensor<T>::from_view(
        x, target_shape, std::move(new_strides), x.offset(), requires_grad, std::move(grad_fn)
    );
}

/**
 * Flatten the tensor. Flatten is a view operation that collapses the range
 * `[start_dim, end_dim]` (inclusive) into a single dimension. Packs first if
 * the source is not contiguous.
 * Example: shape [2,3,4], flatten(1,-1) -> new shape: [2,12]
 *          shape [3,2,3,4], flatten(1,2) -> new shape: [3,6,4]
 *
 * @param x The tensor to flatten.
 * @param start_dim The start dimension (negative indices allowed).
 * @param end_dim The end dimension (negative indices allowed).
 * @return The flattened tensor.
 *
 * @throws std::invalid_argument if the tensor is a scalar.
 * @throws std::invalid_argument if `start_dim > end_dim` after normalization.
 * @throws std::out_of_range if a dimension index is out of range.
 */
template <typename T>
tensor::Tensor<T> flatten(
    const tensor::Tensor<T>& x,
    int64_t start_dim,
    int64_t end_dim) {
    const int64_t rank = x.rank();
    if (rank == 0)
        throw std::invalid_argument("cannot flatten a scalar tensor");

    start_dim = tensor::Tensor<T>::normalize_dimension(start_dim, rank);
    end_dim   = tensor::Tensor<T>::normalize_dimension(end_dim,   rank);

    if (start_dim > end_dim)
        throw std::invalid_argument("flatten: start_dim must be <= end_dim after normalization");

    const tensor::Tensor<T> src = x.is_contiguous() ? x : contiguous(x);

    std::vector<int64_t> new_shape;
    new_shape.reserve(static_cast<size_t>(rank - (end_dim - start_dim)));

    for (int64_t i = 0; i < start_dim; ++i)
        new_shape.push_back(src.shape()[static_cast<size_t>(i)]);

    int64_t flat_size = 1;
    for (int64_t i = start_dim; i <= end_dim; ++i)
        flat_size *= src.shape()[static_cast<size_t>(i)];
    new_shape.push_back(flat_size);

    for (int64_t i = end_dim + 1; i < rank; ++i)
        new_shape.push_back(src.shape()[static_cast<size_t>(i)]);

    const bool requires_grad = autograd::is_grad_enabled() && src.requires_grad();
    std::shared_ptr<autograd::Node<T>> grad_fn = nullptr;
    if (requires_grad)
        grad_fn = std::make_shared<autograd::FlattenBackward<T>>(src, src.shape());
    auto strides = tensor::Tensor<T>::compute_contiguous_strides(new_shape);
    return tensor::Tensor<T>::from_view(
        src, std::move(new_shape), std::move(strides), src.offset(), requires_grad, std::move(grad_fn)
    );
}

/**
 * Squeeze the tensor. Squeeze is a view operation that removes all size-1 dimensions.
 * The stride associated with each removed dimension is dropped; remaining strides are preserved as-is so the
 * op is valid on non-contiguous tensors.
 *
 * @param x The tensor to squeeze.
 * @return The squeezed tensor.
 */
template <typename T>
tensor::Tensor<T> squeeze(const tensor::Tensor<T>& x) {
    std::vector<int64_t> new_shape;
    std::vector<int64_t> new_strides;

    const auto& shape   = x.shape();
    const auto& strides = x.strides();
    // remove all size-1 dimensions
    for (size_t i = 0; i < shape.size(); ++i)
        if (shape[i] != 1) {
            new_shape.push_back(shape[i]);
            new_strides.push_back(strides[i]);
        }

    const bool requires_grad = autograd::is_grad_enabled() && x.requires_grad();
    std::shared_ptr<autograd::Node<T>> grad_fn = nullptr;
    if (requires_grad)
        grad_fn = std::make_shared<autograd::SqueezeBackward<T>>(x, x.shape());
    return tensor::Tensor<T>::from_view(
        x, std::move(new_shape), std::move(new_strides), x.offset(), requires_grad, std::move(grad_fn)
    );
}

/**
 * Squeeze the tensor. Squeeze is a view operation that removes the specified dimension only if its size is 1.
 * If the dimension does not have size 1 the tensor is returned unchanged.
 *
 * @param x The tensor to squeeze.
 * @param dim The dimension to squeeze (negative indices allowed).
 * @return The squeezed tensor.
 *
 * @throws std::out_of_range if `dim` is out of range.
 */
template <typename T>
tensor::Tensor<T> squeeze(const tensor::Tensor<T>& x, int64_t dim) {
    const int64_t rank = x.rank();
    dim = tensor::Tensor<T>::normalize_dimension(dim, rank);

    // Dimension is not size 1; return the same tensor (keep grad_fn / leaf storage).
    if (x.shape()[static_cast<size_t>(dim)] != 1)
        return x;

    std::vector<int64_t> new_shape {x.shape()};
    std::vector<int64_t> new_strides {x.strides()};
    new_shape.erase(new_shape.begin() + dim);
    new_strides.erase(new_strides.begin() + dim);

    const bool requires_grad = autograd::is_grad_enabled() && x.requires_grad();
    std::shared_ptr<autograd::Node<T>> grad_fn = nullptr;
    if (requires_grad)
        grad_fn = std::make_shared<autograd::SqueezeBackward<T>>(x, x.shape());
    return tensor::Tensor<T>::from_view(
        x, std::move(new_shape), std::move(new_strides), x.offset(), requires_grad, std::move(grad_fn)
    );
}

/**
 * Unsqueeze the tensor. 
 * Unsqueeze is a view operation that inserts a new size-1 dimension at the specified dimension.
 * Inserted stride is size[dim] * stride[dim] of the old tensor (1 if inserted at the end), which 
 * is the C-contiguous stride for that axis.
 *
 * @param x The tensor to unsqueeze.
 * @param dim The dimension to unsqueeze.
 * @return The unsqueezed tensor.
 *
 * @throws std::out_of_range if the dimension index is out of range.
 */
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

    new_shape.insert(new_shape.begin() + dim, 1);
    // Since we input dimenion of size 1, the stride formula will receive 0 at that position
    // We need to calculate correct strides so the array is still contiguous.
    // example: shape [2,3]
    // - unsqueeze(0) -> new shape: [1, 2, 3], new strides: [6, 3, 1]
    // - unsqueeze(1) -> new shape: [2, 1, 3], new strides: [3, 3, 1]
    // - unsqueeze(2) -> new shape: [2, 3, 1], new strides: [3, 1, 1]
    const int64_t inserted_stride = (dim < rank)
        ? x.shape()[static_cast<size_t>(dim)] * x.strides()[static_cast<size_t>(dim)]
        : 1;
    new_strides.insert(new_strides.begin() + dim, inserted_stride);

    const bool requires_grad = autograd::is_grad_enabled() && x.requires_grad();
    std::shared_ptr<autograd::Node<T>> grad_fn = nullptr;
    if (requires_grad)
        grad_fn = std::make_shared<autograd::UnsqueezeBackward<T>>(x, x.shape());

    return tensor::Tensor<T>::from_view(
        x, std::move(new_shape), std::move(new_strides), x.offset(), requires_grad, std::move(grad_fn)
    );
}

/**
 * Copy logical elements of `x` into a new dense row-major buffer using
 *   index = offset + sum_d idx[d] * strides[d]
 *
 * @param x The tensor to copy into a contiguous storage.
 * @return The contiguous storage.
 */
template <typename T>
std::vector<T> copy_to_contiguous_storage(const tensor::Tensor<T>& x) {
    const int64_t n = x.numel();
    std::vector<T> out(static_cast<size_t>(n));
    if (n == 0)
        return out;

    logging::info("copying tensor to contiguous storage. numel: " + std::to_string(n));

    const int64_t rank = x.rank();
    const auto& shape = x.shape();
    const auto& strides = x.strides();
    const int64_t offset = x.offset();
    const auto& buf = x.data();
    const auto contiguous_strides = tensor::Tensor<T>::compute_contiguous_strides(shape);

    // `i` is the destination index and `x`'s row-major logical index.
    for (int64_t i = 0; i < n; ++i) {
        int64_t remaining = i;
        int64_t pos = offset;
        for (int64_t d = 0; d < rank; ++d) {
            // Quotient is this coordinate; remainder continues to lower dimensions.
            const int64_t idx = remaining / contiguous_strides[static_cast<size_t>(d)];
            remaining %= contiguous_strides[static_cast<size_t>(d)];
            // Map the coordinate to `x`'s current storage.
            pos += idx * strides[static_cast<size_t>(d)];
        }
        // Copy from source storage index `pos` to contiguous index `i`.
        out[static_cast<size_t>(i)] = buf[static_cast<size_t>(pos)];
    }
    return out;
}

/**
 * Return a tensor whose logical order is packed into a dense row-major buffer.
 * If `x` is already contiguous with `offset() == 0`, returns `x`; otherwise copies
 * via `copy_to_contiguous_storage` and may attach `ContiguousBackward`.
 *
 * @param x The tensor to pack.
 * @return A contiguous tensor with the same shape and logical values.
 */
template <typename T>
tensor::Tensor<T> contiguous(const tensor::Tensor<T>& x) {
    if (x.is_contiguous() && x.offset() == 0)
        return x;

    std::vector<T> storage = copy_to_contiguous_storage(x);

    const bool requires_grad = autograd::is_grad_enabled() && x.requires_grad();
    std::shared_ptr<autograd::Node<T>> grad_fn = nullptr;
    if (requires_grad)
        grad_fn = std::make_shared<autograd::ContiguousBackward<T>>(x);
    return tensor::Tensor<T>::from_operation_result(
        x.shape(), std::move(storage), requires_grad, std::move(grad_fn)
    );
}

/**
 * Narrow the tensor. Narrow is a view operation that returns a view of a slice along `dim` of length `length` starting at `start`.
 *
 * @param x The tensor to narrow.
 * @param dim The dimension to narrow.
 * @param start The start of the slice.
 * @param length The length of the slice.
 * @return The narrowed tensor.
 *
 * @throws std::invalid_argument if the tensor is a scalar.
 * @throws std::invalid_argument if the start/length is out of range.
 * @throws std::out_of_range if `dim` is out of range.
 */
template <typename T>
tensor::Tensor<T> narrow(
    const tensor::Tensor<T>& x,
    int64_t dim,
    int64_t start,
    int64_t length) {
    if (x.rank() == 0)
        throw std::invalid_argument("narrow: input must be non-scalar");

    dim = tensor::Tensor<T>::normalize_dimension(dim, x.rank());
    const int64_t size = x.shape()[static_cast<size_t>(dim)];
    if (start < 0 || length < 0 || start + length > size)
        throw std::invalid_argument("narrow: start/length out of range");

    auto shape = x.shape();
    shape[static_cast<size_t>(dim)] = length;
    const int64_t new_offset = x.offset() + start * x.strides()[static_cast<size_t>(dim)];

    const bool requires_grad = autograd::is_grad_enabled() && x.requires_grad();
    std::shared_ptr<autograd::Node<T>> grad_fn = nullptr;
    if (requires_grad)
        grad_fn = std::make_shared<autograd::NarrowBackward<T>>(x, dim, start);
    return tensor::Tensor<T>::from_view(
        x, std::move(shape), x.strides(), new_offset, requires_grad, std::move(grad_fn));
}

/**
 * Concatenate tensors along `dim`. Inputs are packed and copied into a new
 * contiguous buffer. Attaches `CatBackward` when any input requires grad and
 * grad is enabled.
 *
 * @param tensors Non-empty list of tensors of the same rank.
 * @param dim Dimension to concatenate along (negative indices allowed).
 * @return The concatenated tensor.
 *
 * @throws std::invalid_argument if `tensors` is empty.
 * @throws std::invalid_argument if any tensor is a scalar.
 * @throws std::invalid_argument if the tensors do not all have the same rank.
 * @throws std::invalid_argument if non-`dim` sizes do not match.
 * @throws std::out_of_range if `dim` is out of range.
 */
template <typename T>
tensor::Tensor<T> cat(const std::vector<tensor::Tensor<T>>& tensors, int64_t dim) {
    if (tensors.empty())
        throw std::invalid_argument("cat: expected at least one tensor");

    const int64_t rank = tensors[0].rank();
    if (rank == 0)
        throw std::invalid_argument("cat: cannot concatenate scalars");
    dim = tensor::Tensor<T>::normalize_dimension(dim, rank);

    auto out_shape = tensors[0].shape();
    out_shape[static_cast<size_t>(dim)] = 0;
    bool any_grad = false;
    std::vector<tensor::Tensor<T>> contig;
    contig.reserve(tensors.size());
    for (const auto& t : tensors) {
        if (t.rank() != rank)
            throw std::invalid_argument("cat: all tensors must have the same rank");
        for (int64_t d = 0; d < rank; ++d) {
            if (d == dim)
                continue;
            if (t.shape()[static_cast<size_t>(d)] != out_shape[static_cast<size_t>(d)])
                throw std::invalid_argument("cat: tensor shapes must match except on cat dim");
        }
        out_shape[static_cast<size_t>(dim)] += t.shape()[static_cast<size_t>(dim)];
        any_grad = any_grad || t.requires_grad();
        contig.push_back(contiguous(t));
    }

    const int64_t out_n = tensor::Tensor<T>::compute_numel(out_shape);
    std::vector<T> storage(static_cast<size_t>(out_n));
    const auto out_strides = tensor::Tensor<T>::compute_contiguous_strides(out_shape);

    int64_t dim_offset = 0;
    for (const auto& t : contig) {
        const auto& in_shape = t.shape();
        const auto in_strides = tensor::Tensor<T>::compute_contiguous_strides(in_shape);
        const auto& td = t.data();
        for (int64_t f = 0; f < t.numel(); ++f) {
            int64_t remaining = f;
            int64_t out_f = 0;
            for (int64_t d = 0; d < rank; ++d) {
                int64_t idx = remaining / in_strides[static_cast<size_t>(d)];
                remaining %= in_strides[static_cast<size_t>(d)];
                if (d == dim)
                    idx += dim_offset;
                out_f += idx * out_strides[static_cast<size_t>(d)];
            }
            storage[static_cast<size_t>(out_f)] = td[static_cast<size_t>(f)];
        }
        dim_offset += in_shape[static_cast<size_t>(dim)];
    }

    const bool requires_grad = autograd::is_grad_enabled() && any_grad;
    std::shared_ptr<autograd::Node<T>> grad_fn = nullptr;
    if (requires_grad)
        grad_fn = std::make_shared<autograd::CatBackward<T>>(tensors, dim);

    return tensor::Tensor<T>::from_operation_result(
        out_shape, std::move(storage), requires_grad, std::move(grad_fn));
}

/**
 * Concatenate tensors along `dim`. Builds a vector from the initializer list
 * and delegates to the vector overload.
 *
 * @param tensors Non-empty list of tensors of the same rank.
 * @param dim Dimension to concatenate along (negative indices allowed).
 * @return The concatenated tensor.
 *
 * @throws std::invalid_argument if `tensors` is empty.
 * @throws std::invalid_argument if any tensor is a scalar.
 * @throws std::invalid_argument if the tensors do not all have the same rank.
 * @throws std::invalid_argument if non-`dim` sizes do not match.
 * @throws std::out_of_range if `dim` is out of range.
 */
template <typename T>
tensor::Tensor<T> cat(std::initializer_list<tensor::Tensor<T>> tensors, int64_t dim) {
    return cat(std::vector<tensor::Tensor<T>>(tensors), dim);
}

} // namespace ops
