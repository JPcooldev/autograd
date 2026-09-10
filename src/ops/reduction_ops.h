#pragma once

#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

#include "../tensor/tensor.h"
#include "../autograd/grad_context.h"
#include "../autograd/backward_ops/backward_reduction_ops.h"
#include "shape_ops.h"

namespace ops {

namespace {

struct PackedDim {
    std::vector<int64_t> out_shape;
    int64_t out_numel;
    int64_t outer_size;
    int64_t reduce_size;
    int64_t inner_size;
};

/**
 * Pack the layout of a reduction along `dim`. Builds the output shape (input
 * shape with `dim` removed) and the outer/reduce/inner sizes used to index
 * a contiguous buffer as `outer * reduce * inner + r * inner + i`.
 *
 * @param in_shape Input shape.
 * @param dim Normalized dimension to reduce (must be in range).
 * @return Layout used by dim-reductions and softmax.
 */
template <typename T>
PackedDim packed_dim(const std::vector<int64_t>& in_shape, int64_t dim) {
    const int64_t in_rank = static_cast<int64_t>(in_shape.size());
    PackedDim layout;
    layout.out_shape.reserve(static_cast<size_t>(in_rank - 1));
    for (int64_t d = 0; d < in_rank; ++d)
        if (d != dim)
            layout.out_shape.push_back(in_shape[static_cast<size_t>(d)]);

    layout.out_numel = tensor::Tensor<T>::compute_numel(layout.out_shape);
    layout.reduce_size = in_shape[static_cast<size_t>(dim)];
    layout.inner_size =
        tensor::Tensor<T>::compute_contiguous_strides(in_shape)[static_cast<size_t>(dim)];
    layout.outer_size = layout.inner_size == 0 ? 0 : layout.out_numel / layout.inner_size;
    return layout;
}

} // namespace

/**
 * Sum all elements of a tensor. Packs `x` and reduces to a scalar tensor of
 * shape `{}`. Attaches `SumBackward` when grad is enabled.
 *
 * @param x The tensor to sum.
 * @return A scalar tensor of shape `{}`.
 *
 * @throws std::invalid_argument if `x` is empty.
 */
template <typename T>
tensor::Tensor<T> sum(const tensor::Tensor<T>& x) {
    if (x.numel() == 0)
        throw std::invalid_argument("sum requires a non-empty tensor");

    const auto xc = x.contiguous();
    T total = T{0};
    for (const T& val : xc.data())
        total += val;

    const bool requires_grad = autograd::is_grad_enabled() && x.requires_grad();
    std::shared_ptr<autograd::Node<T>> grad_fn = nullptr;
    if (requires_grad)
        grad_fn = std::make_shared<autograd::SumBackward<T>>(xc);
    return tensor::Tensor<T>::from_operation_result(
        {}, std::vector<T>{total}, requires_grad, std::move(grad_fn));
}

/**
 * Sum along one dimension. Packs `x` and reduces along `dim`; the output shape
 * is the input shape with `dim` removed. Attaches `SumDimBackward` when grad
 * is enabled.
 *
 * @param x The tensor to sum.
 * @param dim Dimension to reduce (negative indices allowed).
 * @return The reduced tensor.
 *
 * @throws std::invalid_argument if `x` is a scalar.
 * @throws std::out_of_range if `dim` is out of range.
 */
template <typename T>
tensor::Tensor<T> sum(const tensor::Tensor<T>& x, int64_t dim) {
    if (x.rank() == 0)
        throw std::invalid_argument("sum(dim) requires a non-scalar tensor");

    dim = tensor::Tensor<T>::normalize_dimension(dim, x.rank());
    const auto xc = x.contiguous();
    const PackedDim layout = packed_dim<T>(xc.shape(), dim);

    std::vector<T> storage(static_cast<size_t>(layout.out_numel), T{0});
    const auto& in_data = xc.data();

    for (int64_t outer = 0; outer < layout.outer_size; ++outer) {
        const int64_t out_base = outer * layout.inner_size;
        for (int64_t r = 0; r < layout.reduce_size; ++r) {
            const int64_t in_base = (outer * layout.reduce_size + r) * layout.inner_size;
            for (int64_t i = 0; i < layout.inner_size; ++i)
                storage[static_cast<size_t>(out_base + i)] +=
                    in_data[static_cast<size_t>(in_base + i)];
        }
    }

    const bool requires_grad = autograd::is_grad_enabled() && x.requires_grad();
    std::shared_ptr<autograd::Node<T>> grad_fn = nullptr;
    if (requires_grad)
        grad_fn = std::make_shared<autograd::SumDimBackward<T>>(xc, dim);
    return tensor::Tensor<T>::from_operation_result(
        layout.out_shape, std::move(storage), requires_grad, std::move(grad_fn));
}

/**
 * Mean of all elements of a tensor. Packs `x`, sums, then divides by `numel`.
 * Attaches `MeanBackward` when grad is enabled.
 *
 * @param x The tensor to average.
 * @return A scalar tensor of shape `{}`.
 *
 * @throws std::invalid_argument if `x` is empty.
 */
template <typename T>
tensor::Tensor<T> mean(const tensor::Tensor<T>& x) {
    if (x.numel() == 0)
        throw std::invalid_argument("mean requires a non-empty tensor");

    const auto xc = contiguous(x);
    T total = T{0};
    for (const T& val : xc.data())
        total += val;

    const T result = total / static_cast<T>(xc.numel());

    const bool requires_grad = autograd::is_grad_enabled() && x.requires_grad();
    std::shared_ptr<autograd::Node<T>> grad_fn = nullptr;
    if (requires_grad)
        grad_fn = std::make_shared<autograd::MeanBackward<T>>(xc);
    return tensor::Tensor<T>::from_operation_result(
        {}, std::vector<T>{result}, requires_grad, std::move(grad_fn));
}

/**
 * Mean along one dimension. Packs `x`, sums along `dim`, then divides by the
 * size of `dim`. Attaches `MeanDimBackward` when grad is enabled.
 *
 * @param x The tensor to average.
 * @param dim Dimension to reduce (negative indices allowed).
 * @return The reduced tensor.
 *
 * @throws std::invalid_argument if `x` is a scalar.
 * @throws std::invalid_argument if `dim` has size 0.
 * @throws std::out_of_range if `dim` is out of range.
 */
template <typename T>
tensor::Tensor<T> mean(const tensor::Tensor<T>& x, int64_t dim) {
    if (x.rank() == 0)
        throw std::invalid_argument("mean(dim) requires a non-scalar tensor");

    dim = tensor::Tensor<T>::normalize_dimension(dim, x.rank());
    const auto xc = contiguous(x);
    const PackedDim layout = packed_dim<T>(xc.shape(), dim);

    if (layout.reduce_size == 0)
        throw std::invalid_argument("mean(dim) cannot reduce over an empty dimension");

    std::vector<T> storage(static_cast<size_t>(layout.out_numel), T{0});
    const auto& in_data = xc.data();

    for (int64_t outer = 0; outer < layout.outer_size; ++outer) {
        const int64_t out_base = outer * layout.inner_size;
        for (int64_t r = 0; r < layout.reduce_size; ++r) {
            const int64_t in_base = (outer * layout.reduce_size + r) * layout.inner_size;
            for (int64_t i = 0; i < layout.inner_size; ++i)
                storage[static_cast<size_t>(out_base + i)] +=
                    in_data[static_cast<size_t>(in_base + i)];
        }
    }

    for (T& val : storage)
        val /= static_cast<T>(layout.reduce_size);

    const bool requires_grad = autograd::is_grad_enabled() && x.requires_grad();
    std::shared_ptr<autograd::Node<T>> grad_fn = nullptr;
    if (requires_grad)
        grad_fn = std::make_shared<autograd::MeanDimBackward<T>>(xc, dim);

    return tensor::Tensor<T>::from_operation_result(
        layout.out_shape, std::move(storage), requires_grad, std::move(grad_fn));
}

/**
 * Maximum of all elements of a tensor. Packs `x` and returns the first
 * maximum as a scalar of shape `{}`. Attaches `MaxBackward` when grad is
 * enabled.
 *
 * @param x The tensor to take the max of.
 * @return A scalar tensor of shape `{}`.
 *
 * @throws std::invalid_argument if `x` is empty.
 */
template <typename T>
tensor::Tensor<T> max(const tensor::Tensor<T>& x) {
    if (x.numel() == 0)
        throw std::invalid_argument("max requires a non-empty tensor");

    const auto xc = contiguous(x);
    const auto& data = xc.data();
    int64_t argmax = 0;
    T max_val = data[0];
    for (int64_t i = 1; i < xc.numel(); ++i) {
        if (data[static_cast<size_t>(i)] > max_val) {
            max_val = data[static_cast<size_t>(i)];
            argmax  = i;
        }
    }

    const bool requires_grad = autograd::is_grad_enabled() && x.requires_grad();
    std::shared_ptr<autograd::Node<T>> grad_fn = nullptr;
    if (requires_grad)
        grad_fn = std::make_shared<autograd::MaxBackward<T>>(xc, argmax);
    return tensor::Tensor<T>::from_operation_result(
        {}, std::vector<T>{max_val}, requires_grad, std::move(grad_fn));
}

/**
 * Minimum of all elements of a tensor. Packs `x` and returns the first
 * minimum as a scalar of shape `{}`. Attaches `MinBackward` when grad is
 * enabled.
 *
 * @param x The tensor to take the min of.
 * @return A scalar tensor of shape `{}`.
 *
 * @throws std::invalid_argument if `x` is empty.
 */
template <typename T>
tensor::Tensor<T> min(const tensor::Tensor<T>& x) {
    if (x.numel() == 0)
        throw std::invalid_argument("min requires a non-empty tensor");

    const auto xc = contiguous(x);
    const auto& data = xc.data();
    int64_t argmin = 0;
    T min_val = data[0];
    for (int64_t i = 1; i < xc.numel(); ++i) {
        if (data[static_cast<size_t>(i)] < min_val) {
            min_val = data[static_cast<size_t>(i)];
            argmin  = i;
        }
    }
    const bool requires_grad = autograd::is_grad_enabled() && x.requires_grad();
    std::shared_ptr<autograd::Node<T>> grad_fn = nullptr;
    if (requires_grad)
        grad_fn = std::make_shared<autograd::MinBackward<T>>(xc, argmin);

    return tensor::Tensor<T>::from_operation_result(
        {}, std::vector<T>{min_val}, requires_grad, std::move(grad_fn));
}

/**
 * Maximum along one dimension. Packs `x` and reduces along `dim`; ties keep
 * the first occurrence. Attaches `MaxDimBackward` when grad is enabled.
 *
 * @param x The tensor to take the max of.
 * @param dim Dimension to reduce (negative indices allowed).
 * @return The reduced tensor.
 *
 * @throws std::invalid_argument if `x` is a scalar.
 * @throws std::invalid_argument if `dim` has size 0.
 * @throws std::out_of_range if `dim` is out of range.
 */
template <typename T>
tensor::Tensor<T> max(const tensor::Tensor<T>& x, int64_t dim) {
    if (x.rank() == 0)
        throw std::invalid_argument("max(dim) requires a non-scalar tensor");

    dim = tensor::Tensor<T>::normalize_dimension(dim, x.rank());
    const auto xc = contiguous(x);
    const PackedDim layout = packed_dim<T>(xc.shape(), dim);

    if (layout.reduce_size == 0)
        throw std::invalid_argument("max(dim) cannot reduce over an empty dimension");

    std::vector<T> storage(static_cast<size_t>(layout.out_numel));
    std::vector<int64_t> argmax(static_cast<size_t>(layout.out_numel));
    const auto& in_data = xc.data();

    for (int64_t outer = 0; outer < layout.outer_size; ++outer) {
        const int64_t out_base = outer * layout.inner_size;
        const int64_t in0 = outer * layout.reduce_size * layout.inner_size;
        for (int64_t i = 0; i < layout.inner_size; ++i) {
            storage[static_cast<size_t>(out_base + i)] = in_data[static_cast<size_t>(in0 + i)];
            argmax[static_cast<size_t>(out_base + i)] = in0 + i;
        }
        for (int64_t r = 1; r < layout.reduce_size; ++r) {
            const int64_t in_base = (outer * layout.reduce_size + r) * layout.inner_size;
            for (int64_t i = 0; i < layout.inner_size; ++i) {
                const T val = in_data[static_cast<size_t>(in_base + i)];
                if (val > storage[static_cast<size_t>(out_base + i)]) {
                    storage[static_cast<size_t>(out_base + i)] = val;
                    argmax[static_cast<size_t>(out_base + i)] = in_base + i;
                }
            }
        }
    }

    const bool requires_grad = autograd::is_grad_enabled() && x.requires_grad();
    std::shared_ptr<autograd::Node<T>> grad_fn = nullptr;
    if (requires_grad)
        grad_fn = std::make_shared<autograd::MaxDimBackward<T>>(xc, std::move(argmax));
    return tensor::Tensor<T>::from_operation_result(
        layout.out_shape, std::move(storage), requires_grad, std::move(grad_fn));
}

/**
 * Minimum along one dimension. Packs `x` and reduces along `dim`; ties keep
 * the first occurrence. Attaches `MinDimBackward` when grad is enabled.
 *
 * @param x The tensor to take the min of.
 * @param dim Dimension to reduce (negative indices allowed).
 * @return The reduced tensor.
 *
 * @throws std::invalid_argument if `x` is a scalar.
 * @throws std::invalid_argument if `dim` has size 0.
 * @throws std::out_of_range if `dim` is out of range.
 */
template <typename T>
tensor::Tensor<T> min(const tensor::Tensor<T>& x, int64_t dim) {
    if (x.rank() == 0)
        throw std::invalid_argument("min(dim) requires a non-scalar tensor");

    dim = tensor::Tensor<T>::normalize_dimension(dim, x.rank());
    const auto xc = contiguous(x);
    const PackedDim layout = packed_dim<T>(xc.shape(), dim);

    if (layout.reduce_size == 0)
        throw std::invalid_argument("min(dim) cannot reduce over an empty dimension");

    std::vector<T> storage(static_cast<size_t>(layout.out_numel));
    std::vector<int64_t> argmin(static_cast<size_t>(layout.out_numel));
    const auto& in_data = xc.data();

    for (int64_t outer = 0; outer < layout.outer_size; ++outer) {
        const int64_t out_base = outer * layout.inner_size;
        const int64_t in0 = outer * layout.reduce_size * layout.inner_size;
        for (int64_t i = 0; i < layout.inner_size; ++i) {
            storage[static_cast<size_t>(out_base + i)] = in_data[static_cast<size_t>(in0 + i)];
            argmin[static_cast<size_t>(out_base + i)] = in0 + i;
        }
        for (int64_t r = 1; r < layout.reduce_size; ++r) {
            const int64_t in_base = (outer * layout.reduce_size + r) * layout.inner_size;
            for (int64_t i = 0; i < layout.inner_size; ++i) {
                const T val = in_data[static_cast<size_t>(in_base + i)];
                if (val < storage[static_cast<size_t>(out_base + i)]) {
                    storage[static_cast<size_t>(out_base + i)] = val;
                    argmin[static_cast<size_t>(out_base + i)] = in_base + i;
                }
            }
        }
    }

    const bool requires_grad = autograd::is_grad_enabled() && x.requires_grad();
    std::shared_ptr<autograd::Node<T>> grad_fn = nullptr;
    if (requires_grad)
        grad_fn = std::make_shared<autograd::MinDimBackward<T>>(xc, std::move(argmin));
    return tensor::Tensor<T>::from_operation_result(
        layout.out_shape, std::move(storage), requires_grad, std::move(grad_fn));
}

/**
 * Softmax along one dimension. Packs `x`, subtracts the per-slice max, then
 * exponentiates and normalizes. Output has the same shape as `x`. Attaches
 * `SoftmaxBackward` when grad is enabled.
 *
 * @param x The tensor to softmax.
 * @param dim Dimension to softmax over (negative indices allowed).
 * @return A tensor with the same shape as `x`.
 *
 * @throws std::invalid_argument if `x` is a scalar.
 * @throws std::invalid_argument if `dim` has size 0.
 * @throws std::out_of_range if `dim` is out of range.
 */
template <typename T>
tensor::Tensor<T> softmax(const tensor::Tensor<T>& x, int64_t dim) {
    if (x.rank() == 0)
        throw std::invalid_argument("softmax requires a non-scalar tensor");

    dim = tensor::Tensor<T>::normalize_dimension(dim, x.rank());
    const auto xc = contiguous(x);
    const PackedDim layout = packed_dim<T>(xc.shape(), dim);

    if (layout.reduce_size == 0)
        throw std::invalid_argument("softmax cannot reduce over an empty dimension");

    const auto& in_data = xc.data();
    const auto& in_shape = xc.shape();
    const int64_t n = xc.numel();

    std::vector<T> slice_max(
        static_cast<size_t>(layout.out_numel), std::numeric_limits<T>::lowest());
    for (int64_t outer = 0; outer < layout.outer_size; ++outer) {
        const int64_t out_base = outer * layout.inner_size;
        for (int64_t r = 0; r < layout.reduce_size; ++r) {
            const int64_t in_base = (outer * layout.reduce_size + r) * layout.inner_size;
            for (int64_t i = 0; i < layout.inner_size; ++i)
                slice_max[static_cast<size_t>(out_base + i)] = std::max(
                    slice_max[static_cast<size_t>(out_base + i)],
                    in_data[static_cast<size_t>(in_base + i)]);
        }
    }

    std::vector<T> storage(static_cast<size_t>(n));
    std::vector<T> slice_sum(static_cast<size_t>(layout.out_numel), T{0});
    for (int64_t outer = 0; outer < layout.outer_size; ++outer) {
        const int64_t out_base = outer * layout.inner_size;
        for (int64_t r = 0; r < layout.reduce_size; ++r) {
            const int64_t in_base = (outer * layout.reduce_size + r) * layout.inner_size;
            for (int64_t i = 0; i < layout.inner_size; ++i) {
                const size_t in_i = static_cast<size_t>(in_base + i);
                const size_t out_i = static_cast<size_t>(out_base + i);
                storage[in_i] = static_cast<T>(std::exp(in_data[in_i] - slice_max[out_i]));
                slice_sum[out_i] += storage[in_i];
            }
        }
    }

    for (int64_t outer = 0; outer < layout.outer_size; ++outer) {
        const int64_t out_base = outer * layout.inner_size;
        for (int64_t r = 0; r < layout.reduce_size; ++r) {
            const int64_t in_base = (outer * layout.reduce_size + r) * layout.inner_size;
            for (int64_t i = 0; i < layout.inner_size; ++i)
                storage[static_cast<size_t>(in_base + i)] /=
                    slice_sum[static_cast<size_t>(out_base + i)];
        }
    }

    const bool requires_grad = autograd::is_grad_enabled() && x.requires_grad();
    std::shared_ptr<autograd::Node<T>> grad_fn = nullptr;
    if (requires_grad) {
        const tensor::Tensor<T> s_saved = tensor::Tensor<T>::from_operation_result(
            in_shape, std::vector<T>(storage), false, nullptr);
        grad_fn = std::make_shared<autograd::SoftmaxBackward<T>>(xc, s_saved, dim);
    }
    return tensor::Tensor<T>::from_operation_result(
        in_shape, std::move(storage), requires_grad, std::move(grad_fn));
}

} // namespace ops
