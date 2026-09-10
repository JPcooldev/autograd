#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "../node.h"

namespace autograd {

using tensor::Tensor;

namespace {

struct PackedDim {
    std::vector<int64_t> out_shape;
    int64_t out_numel;
    int64_t outer_size;
    int64_t reduce_size;
    int64_t inner_size;
};

/**
 * Pack a reduction axis into outer / reduce / inner sizes for a contiguous tensor.
 * `out_shape` is `in_shape` with `dim` removed.
 *
 * @param in_shape Input tensor shape.
 * @param dim Reduction dimension (already normalized, in `[0, rank)`).
 * @return Layout used by dim-wise backward kernels.
 *
 * @throws std::invalid_argument if `in_shape` has a negative dimension.
 * @throws std::overflow_error if a numel product overflows.
 */
template <typename T>
PackedDim packed_dim(const std::vector<int64_t>& in_shape, int64_t dim) {
    const int64_t in_rank = static_cast<int64_t>(in_shape.size());
    PackedDim layout;
    layout.out_shape.reserve(static_cast<size_t>(in_rank - 1));
    for (int64_t d = 0; d < in_rank; ++d)
        if (d != dim)
            layout.out_shape.push_back(in_shape[static_cast<size_t>(d)]);

    layout.out_numel = Tensor<T>::compute_numel(layout.out_shape);
    layout.reduce_size = in_shape[static_cast<size_t>(dim)];
    layout.inner_size =
        Tensor<T>::compute_contiguous_strides(in_shape)[static_cast<size_t>(dim)];
    layout.outer_size = layout.inner_size == 0 ? 0 : layout.out_numel / layout.inner_size;
    return layout;
}

} // namespace

template <typename T>
class SumBackward : public Node<T> {
    std::vector<int64_t> input_shape_;

public:
    /**
     * Construct the backward node for a full sum-to-scalar.
     * Aliases `x` for the graph edge and stores the input shape.
     *
     * @param x Forward input (edge only; unused in apply).
     */
    explicit SumBackward(const Tensor<T>& x)
        : Node<T>(x), input_shape_(x.shape()) {}

    /**
     * Broadcast a scalar gradient to every input element.
     * Uses stored `input_shape_`; saved tensor values are unused.
     *
     * @param propagated_grad Scalar upstream gradient dL/d(sum).
     * @return `{dL/dx}` filled with `propagated_grad.data()[0]`.
     */
    std::vector<Tensor<T>> apply(const Tensor<T>& propagated_grad) override {
        const T grad_val = propagated_grad.data()[0];
        const int64_t numel = Tensor<T>::compute_numel(input_shape_);
        std::vector<T> out(static_cast<size_t>(numel), grad_val);
        return {Tensor<T>::from_operation_result(input_shape_, std::move(out), false, nullptr)};
    }
};

template <typename T>
class SumDimBackward : public Node<T> {
    std::vector<int64_t> input_shape_;
    int64_t dim_;

public:
    /**
     * Construct the backward node for sum along one dimension.
     * Aliases `x` and stores the input shape and reduction dim.
     *
     * @param x Forward input (edge only; unused in apply).
     * @param dim Dimension that was reduced.
     */
    SumDimBackward(const Tensor<T>& x, int64_t dim)
        : Node<T>(x), input_shape_(x.shape()), dim_(dim) {}

    /**
     * Expand `propagated_grad` along the reduced dim (copy each output value
     * onto every position in that slice). Uses `input_shape_` and `dim_`.
     *
     * @param propagated_grad Upstream gradient with the reduced shape.
     * @return `{dL/dx}` with the original input shape.
     *
     * @throws std::invalid_argument if a stored shape has a negative dimension.
     * @throws std::overflow_error if a numel product overflows.
     */
    std::vector<Tensor<T>> apply(const Tensor<T>& propagated_grad) override {
        const PackedDim layout = packed_dim<T>(input_shape_, dim_);
        const int64_t in_numel = Tensor<T>::compute_numel(input_shape_);
        std::vector<T> grad_in(static_cast<size_t>(in_numel));
        const auto& g = propagated_grad.data();

        for (int64_t outer = 0; outer < layout.outer_size; ++outer) {
            const int64_t out_base = outer * layout.inner_size;
            for (int64_t r = 0; r < layout.reduce_size; ++r) {
                const int64_t in_base = (outer * layout.reduce_size + r) * layout.inner_size;
                for (int64_t i = 0; i < layout.inner_size; ++i)
                    grad_in[static_cast<size_t>(in_base + i)] = g[static_cast<size_t>(out_base + i)];
            }
        }
        return {Tensor<T>::from_operation_result(input_shape_, std::move(grad_in), false, nullptr)};
    }
};

template <typename T>
class MeanBackward : public Node<T> {
    std::vector<int64_t> input_shape_;
    int64_t numel_;

public:
    /**
     * Construct the backward node for a full mean-to-scalar.
     * Aliases `x` and stores the input shape and numel.
     *
     * @param x Forward input (edge only; unused in apply).
     */
    explicit MeanBackward(const Tensor<T>& x)
        : Node<T>(x), input_shape_(x.shape()), numel_(x.numel()) {}

    /**
     * Broadcast `propagated_grad / numel` to every input element.
     * Uses stored `input_shape_` and `numel_`.
     *
     * @param propagated_grad Scalar upstream gradient dL/d(mean).
     * @return `{dL/dx}`.
     */
    std::vector<Tensor<T>> apply(const Tensor<T>& propagated_grad) override {
        const T grad_val = propagated_grad.data()[0] / static_cast<T>(numel_);
        std::vector<T> out(static_cast<size_t>(numel_), grad_val);
        return {Tensor<T>::from_operation_result(input_shape_, std::move(out), false, nullptr)};
    }
};

template <typename T>
class MeanDimBackward : public Node<T> {
    std::vector<int64_t> input_shape_;
    int64_t dim_;
    int64_t dim_size_;

public:
    /**
     * Construct the backward node for mean along one dimension.
     * Aliases `x` and stores input shape, dim, and that dim's size.
     *
     * @param x Forward input (edge only; unused in apply).
     * @param dim Dimension that was reduced.
     */
    MeanDimBackward(const Tensor<T>& x, int64_t dim)
        : Node<T>(x), input_shape_(x.shape()), dim_(dim),
          dim_size_(x.shape()[static_cast<size_t>(dim)]) {}

    /**
     * Expand `propagated_grad` along `dim` and scale by `1 / dim_size_`.
     *
     * @param propagated_grad Upstream gradient with the reduced shape.
     * @return `{dL/dx}` with the original input shape.
     *
     * @throws std::invalid_argument if a stored shape has a negative dimension.
     * @throws std::overflow_error if a numel product overflows.
     */
    std::vector<Tensor<T>> apply(const Tensor<T>& propagated_grad) override {
        const PackedDim layout = packed_dim<T>(input_shape_, dim_);
        const int64_t in_numel = Tensor<T>::compute_numel(input_shape_);
        std::vector<T> grad_in(static_cast<size_t>(in_numel));
        const auto& g = propagated_grad.data();
        const T scale = static_cast<T>(1) / static_cast<T>(dim_size_);

        for (int64_t outer = 0; outer < layout.outer_size; ++outer) {
            const int64_t out_base = outer * layout.inner_size;
            for (int64_t r = 0; r < layout.reduce_size; ++r) {
                const int64_t in_base = (outer * layout.reduce_size + r) * layout.inner_size;
                for (int64_t i = 0; i < layout.inner_size; ++i)
                    grad_in[static_cast<size_t>(in_base + i)] =
                        g[static_cast<size_t>(out_base + i)] * scale;
            }
        }
        return {Tensor<T>::from_operation_result(input_shape_, std::move(grad_in), false, nullptr)};
    }
};

template <typename T>
class MaxBackward : public Node<T> {
    std::vector<int64_t> input_shape_;
    int64_t argmax_;

public:
    /**
     * Construct the backward node for a full max-to-scalar.
     * Aliases `x` and stores the input shape and flat argmax index.
     *
     * @param x Forward input (edge only; unused in apply).
     * @param argmax Flat index of the first maximum (tie-break).
     */
    MaxBackward(const Tensor<T>& x, int64_t argmax)
        : Node<T>(x), input_shape_(x.shape()), argmax_(argmax) {}

    /**
     * Place the scalar gradient at `argmax_` and zeros elsewhere.
     *
     * @param propagated_grad Scalar upstream gradient dL/d(max).
     * @return `{dL/dx}`.
     */
    std::vector<Tensor<T>> apply(const Tensor<T>& propagated_grad) override {
        const int64_t numel = Tensor<T>::compute_numel(input_shape_);
        std::vector<T> out(static_cast<size_t>(numel), T{0});
        out[static_cast<size_t>(argmax_)] = propagated_grad.data()[0];
        return {Tensor<T>::from_operation_result(input_shape_, std::move(out), false, nullptr)};
    }
};

template <typename T>
class MaxDimBackward : public Node<T> {
    std::vector<int64_t> input_shape_;
    std::vector<int64_t> argmax_;

public:
    /**
     * Construct the backward node for max along one dimension.
     * Aliases `x` and stores the input shape and per-slice argmax indices.
     *
     * @param x Forward input (edge only; unused in apply).
     * @param argmax Flat input index of the max in each reduced slice.
     */
    MaxDimBackward(const Tensor<T>& x, std::vector<int64_t> argmax)
        : Node<T>(x), input_shape_(x.shape()), argmax_(std::move(argmax)) {}

    /**
     * Scatter `propagated_grad` onto the stored argmax indices.
     *
     * @param propagated_grad Upstream gradient with the reduced shape.
     * @return `{dL/dx}` with zeros except at argmax positions.
     */
    std::vector<Tensor<T>> apply(const Tensor<T>& propagated_grad) override {
        const int64_t numel = Tensor<T>::compute_numel(input_shape_);
        std::vector<T> out(static_cast<size_t>(numel), T{0});
        const auto& g = propagated_grad.data();
        for (size_t i = 0; i < argmax_.size(); ++i)
            out[static_cast<size_t>(argmax_[i])] = g[i];
        return {Tensor<T>::from_operation_result(input_shape_, std::move(out), false, nullptr)};
    }
};

template <typename T>
class MinBackward : public Node<T> {
    std::vector<int64_t> input_shape_;
    int64_t argmin_;

public:
    /**
     * Construct the backward node for a full min-to-scalar.
     * Aliases `x` and stores the input shape and flat argmin index.
     *
     * @param x Forward input (edge only; unused in apply).
     * @param argmin Flat index of the first minimum (tie-break).
     */
    MinBackward(const Tensor<T>& x, int64_t argmin)
        : Node<T>(x), input_shape_(x.shape()), argmin_(argmin) {}

    /**
     * Place the scalar gradient at `argmin_` and zeros elsewhere.
     *
     * @param propagated_grad Scalar upstream gradient dL/d(min).
     * @return `{dL/dx}`.
     */
    std::vector<Tensor<T>> apply(const Tensor<T>& propagated_grad) override {
        const int64_t numel = Tensor<T>::compute_numel(input_shape_);
        std::vector<T> out(static_cast<size_t>(numel), T{0});
        out[static_cast<size_t>(argmin_)] = propagated_grad.data()[0];
        return {Tensor<T>::from_operation_result(input_shape_, std::move(out), false, nullptr)};
    }
};

template <typename T>
class MinDimBackward : public Node<T> {
    std::vector<int64_t> input_shape_;
    std::vector<int64_t> argmin_;

public:
    /**
     * Construct the backward node for min along one dimension.
     * Aliases `x` and stores the input shape and per-slice argmin indices.
     *
     * @param x Forward input (edge only; unused in apply).
     * @param argmin Flat input index of the min in each reduced slice.
     */
    MinDimBackward(const Tensor<T>& x, std::vector<int64_t> argmin)
        : Node<T>(x), input_shape_(x.shape()), argmin_(std::move(argmin)) {}

    /**
     * Scatter `propagated_grad` onto the stored argmin indices.
     *
     * @param propagated_grad Upstream gradient with the reduced shape.
     * @return `{dL/dx}` with zeros except at argmin positions.
     */
    std::vector<Tensor<T>> apply(const Tensor<T>& propagated_grad) override {
        const int64_t numel = Tensor<T>::compute_numel(input_shape_);
        std::vector<T> out(static_cast<size_t>(numel), T{0});
        const auto& g = propagated_grad.data();
        for (size_t i = 0; i < argmin_.size(); ++i)
            out[static_cast<size_t>(argmin_[i])] = g[i];
        return {Tensor<T>::from_operation_result(input_shape_, std::move(out), false, nullptr)};
    }
};

template <typename T>
class SoftmaxBackward : public Node<T> {
    Tensor<T> output_;   // saved softmax output s
    int64_t   dim_;      // normalized reduction dim (stored pre-normalized)

public:
    /**
     * Construct the backward node for softmax along `dim`.
     * Aliases `x` for the graph edge and aliases softmax output `s` into `output_`.
     *
     * @param x Forward logits (edge only; unused in apply).
     * @param s Forward softmax output (same shape as `x`).
     * @param dim Dimension softmax was taken over (already normalized).
     */
    SoftmaxBackward(const Tensor<T>& x, const Tensor<T>& s, int64_t dim)
        : Node<T>(x), output_(Tensor<T>::alias(s)), dim_(dim) {}

    /**
     * Compute softmax Jacobian-vector product: dx = s * (g - sum(g * s, dim)).
     * Uses saved `output_` and `dim_`; saved input `x` is unused.
     *
     * @param propagated_grad Upstream gradient dL/ds, same shape as `s`.
     * @return `{dL/dx}`.
     *
     * @throws std::invalid_argument if a stored shape has a negative dimension.
     * @throws std::overflow_error if a numel product overflows.
     */
    std::vector<Tensor<T>> apply(const Tensor<T>& propagated_grad) override {
        const Tensor<T>& s = output_;
        const PackedDim layout = packed_dim<T>(s.shape(), dim_);
        const auto& s_data = s.data();
        const auto& g = propagated_grad.data();

        std::vector<T> dot_gs(static_cast<size_t>(layout.out_numel), T{0});
        for (int64_t outer = 0; outer < layout.outer_size; ++outer) {
            const int64_t out_base = outer * layout.inner_size;
            for (int64_t r = 0; r < layout.reduce_size; ++r) {
                const int64_t in_base = (outer * layout.reduce_size + r) * layout.inner_size;
                for (int64_t i = 0; i < layout.inner_size; ++i)
                    dot_gs[static_cast<size_t>(out_base + i)] +=
                        g[static_cast<size_t>(in_base + i)] *
                        s_data[static_cast<size_t>(in_base + i)];
            }
        }

        std::vector<T> grad_storage(static_cast<size_t>(s.numel()));
        for (int64_t outer = 0; outer < layout.outer_size; ++outer) {
            const int64_t out_base = outer * layout.inner_size;
            for (int64_t r = 0; r < layout.reduce_size; ++r) {
                const int64_t in_base = (outer * layout.reduce_size + r) * layout.inner_size;
                for (int64_t i = 0; i < layout.inner_size; ++i) {
                    const size_t in_i = static_cast<size_t>(in_base + i);
                    grad_storage[in_i] = s_data[in_i] *
                        (g[in_i] - dot_gs[static_cast<size_t>(out_base + i)]);
                }
            }
        }
        return {Tensor<T>::from_operation_result(s.shape(), std::move(grad_storage), false, nullptr)};
    }
};

} // namespace autograd
