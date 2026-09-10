#pragma once

#include <memory>
#include <vector>

#include "../engine.h"
#include "../node.h"

namespace autograd {

using tensor::Tensor;

template <typename T>
class TransposeBackward : public Node<T> {
    int64_t dim0_;
    int64_t dim1_;

public:
    /**
     * Construct the backward node for transpose.
     * Aliases `x` for the graph edge and stores the swapped dimension pair.
     *
     * @param x Forward input (edge only; unused in apply).
     * @param dim0 First transposed axis (already normalized).
     * @param dim1 Second transposed axis (already normalized).
     */
    TransposeBackward(const Tensor<T>& x, const int64_t dim0, const int64_t dim1)
        : Node<T>(x), dim0_(dim0), dim1_(dim1) {}

    /**
     * Invert a transpose by transposing `propagated_grad` on the same axes.
     * Uses stored `dim0_` and `dim1_`.
     *
     * @param propagated_grad Upstream gradient dL/d(transposed output).
     * @return `{dL/dx}`.
     *
     * @throws std::invalid_argument if `propagated_grad` is a scalar.
     * @throws std::out_of_range if `dim0_` or `dim1_` is out of range for
     *         `propagated_grad`.
     */
    std::vector<Tensor<T>> apply(const Tensor<T>& propagated_grad) override {
        return {propagated_grad.transpose(dim0_, dim1_)};
    }
};

/**
 * Allocate a `TransposeBackward` node for `input`.
 *
 * @param input Forward tensor that was transposed.
 * @param dim0 First transposed axis.
 * @param dim1 Second transposed axis.
 * @return Shared pointer to the new node.
 */
template <typename T>
std::shared_ptr<Node<T>> make_transpose_backward_node(const Tensor<T>& input,
                                                      const int64_t dim0,
                                                      const int64_t dim1) {
    return std::make_shared<TransposeBackward<T>>(input, dim0, dim1);
}

template <typename T>
class ReshapeBackward : public Node<T> {
    std::vector<int64_t> original_shape_;

public:
    /**
     * Construct the backward node for reshape.
     * Aliases `x` for the graph edge and stores the pre-reshape shape.
     *
     * @param x Forward input (edge only; unused in apply).
     * @param original_shape Shape of `x` before reshape.
     */
    ReshapeBackward(const Tensor<T>& x, std::vector<int64_t> original_shape)
        : Node<T>(x), original_shape_(std::move(original_shape)) {}

    /**
     * Reshape `propagated_grad` back to `original_shape_`.
     *
     * @param propagated_grad Upstream gradient dL/d(reshaped output).
     * @return `{dL/dx}`.
     *
     * @throws std::invalid_argument if `propagated_grad` is a scalar.
     * @throws std::invalid_argument if numel would change.
     */
    std::vector<Tensor<T>> apply(const Tensor<T>& propagated_grad) override {
        return {propagated_grad.reshape(original_shape_)};
    }
};

/**
 * Sum `grad` back to `target_shape` over broadcast axes.
 * Sums prepended leading dims and axes where `target_shape[i] == 1` but
 * `grad` is larger. Assumes `grad` is contiguous (flat index = logical index).
 *
 * @param grad Gradient in the broadcast (output) shape.
 * @param target_shape Shape of the tensor before `broadcast_to`.
 * @return Gradient with `target_shape`.
 *
 * @throws std::invalid_argument if `target_shape` has a negative dimension.
 * @throws std::overflow_error if a numel product overflows.
 */
template <typename T>
Tensor<T> sum_to(const Tensor<T>& grad, const std::vector<int64_t>& target_shape) {
    if (grad.shape() == target_shape)
        return Tensor<T>::from_operation_result(
            target_shape,
            std::vector<T>(grad.data().begin(), grad.data().end()),
            false, nullptr
        );

    const int64_t grad_rank   = grad.rank();
    const int64_t target_rank = static_cast<int64_t>(target_shape.size());
    const int64_t rank_diff   = grad_rank - target_rank;

    const auto grad_cont_strides = Tensor<T>::compute_contiguous_strides(grad.shape());
    const auto out_cont_strides  = Tensor<T>::compute_contiguous_strides(target_shape);

    const int64_t out_numel = Tensor<T>::compute_numel(target_shape);
    std::vector<T> out(static_cast<size_t>(out_numel), T{0});

    for (int64_t flat = 0; flat < grad.numel(); ++flat) {
        int64_t out_flat  = 0;
        int64_t remaining = flat;

        for (int64_t d = 0; d < grad_rank; ++d) {
            const int64_t idx = remaining / grad_cont_strides[static_cast<size_t>(d)];
            remaining        %= grad_cont_strides[static_cast<size_t>(d)];

            const int64_t td = d - rank_diff;

            const bool is_broadcast_dim =
                (td < 0) || (target_shape[static_cast<size_t>(td)] == 1);

            if (!is_broadcast_dim)
                out_flat += idx * out_cont_strides[static_cast<size_t>(td)];
        }

        out[static_cast<size_t>(out_flat)] += grad.data()[static_cast<size_t>(flat)];
    }

    return Tensor<T>::from_operation_result(
        target_shape, std::move(out), false, nullptr
    );
}

template <typename T>
class FlattenBackward : public Node<T> {
    std::vector<int64_t> original_shape_;

public:
    /**
     * Construct the backward node for flatten.
     * Aliases `x` and stores the pre-flatten shape.
     *
     * @param x Forward input (edge only; unused in apply).
     * @param original_shape Shape of `x` before flatten.
     */
    FlattenBackward(const Tensor<T>& x, std::vector<int64_t> original_shape)
        : Node<T>(x), original_shape_(std::move(original_shape)) {}

    /**
     * Reshape `propagated_grad` back to `original_shape_`.
     *
     * @param propagated_grad Upstream gradient dL/d(flattened output).
     * @return `{dL/dx}`.
     *
     * @throws std::invalid_argument if `propagated_grad` is a scalar.
     * @throws std::invalid_argument if numel would change.
     */
    std::vector<Tensor<T>> apply(const Tensor<T>& propagated_grad) override {
        return {propagated_grad.reshape(original_shape_)};
    }
};

template <typename T>
class SqueezeBackward : public Node<T> {
    std::vector<int64_t> original_shape_;

public:
    /**
     * Construct the backward node for squeeze (all axes or a single dim).
     * Aliases `x` and stores the pre-squeeze shape.
     *
     * @param x Forward input (edge only; unused in apply).
     * @param original_shape Shape of `x` before squeeze.
     */
    SqueezeBackward(const Tensor<T>& x, std::vector<int64_t> original_shape)
        : Node<T>(x), original_shape_(std::move(original_shape)) {}

    /**
     * Reshape `propagated_grad` back to `original_shape_`.
     *
     * @param propagated_grad Upstream gradient dL/d(squeezed output).
     * @return `{dL/dx}`.
     *
     * @throws std::invalid_argument if `propagated_grad` is a scalar.
     * @throws std::invalid_argument if numel would change.
     */
    std::vector<Tensor<T>> apply(const Tensor<T>& propagated_grad) override {
        // squeeze of an all-ones shape yields a rank-0 scalar; reshape rejects
        // rank-0 inputs, so rebuild the original layout from the scalar value.
        if (propagated_grad.rank() == 0) {
            const int64_t n = Tensor<T>::compute_numel(original_shape_);
            std::vector<T> storage(static_cast<size_t>(n), propagated_grad.data()[0]);
            return {Tensor<T>::from_operation_result(
                original_shape_, std::move(storage), false, nullptr)};
        }
        return {propagated_grad.reshape(original_shape_)};
    }
};

template <typename T>
class UnsqueezeBackward : public Node<T> {
    std::vector<int64_t> original_shape_;

public:
    /**
     * Construct the backward node for unsqueeze.
     * Aliases `x` and stores the pre-unsqueeze shape.
     *
     * @param x Forward input (edge only; unused in apply).
     * @param original_shape Shape of `x` before unsqueeze.
     */
    UnsqueezeBackward(const Tensor<T>& x, std::vector<int64_t> original_shape)
        : Node<T>(x), original_shape_(std::move(original_shape)) {}

    /**
     * Reshape `propagated_grad` back to `original_shape_`.
     *
     * @param propagated_grad Upstream gradient dL/d(unsqueezed output).
     * @return `{dL/dx}`.
     *
     * @throws std::invalid_argument if `propagated_grad` is a scalar.
     * @throws std::invalid_argument if numel would change.
     */
    std::vector<Tensor<T>> apply(const Tensor<T>& propagated_grad) override {
        return {propagated_grad.reshape(original_shape_)};
    }
};

template <typename T>
class ContiguousBackward : public Node<T> {
public:
    /**
     * Construct the backward node for contiguous.
     * Aliases `x` for the graph edge.
     *
     * @param x Forward input (edge only; unused in apply).
     */
    explicit ContiguousBackward(const Tensor<T>& x) : Node<T>(x) {}

    /**
     * Identity backward: layout-only op, logical values are unchanged.
     *
     * @param propagated_grad Upstream gradient dL/d(contiguous output).
     * @return `{propagated_grad}` unchanged.
     */
    std::vector<Tensor<T>> apply(const Tensor<T>& propagated_grad) override {
        return {propagated_grad};
    }
};

template <typename T>
class BroadcastToBackward : public Node<T> {
    std::vector<int64_t> original_shape_;

public:
    /**
     * Construct the backward node for broadcast_to.
     * Aliases `x` for the graph edge and stores the pre-broadcast shape.
     *
     * @param x Forward input (edge only; unused in apply).
     * @param original_shape Shape of `x` before broadcast.
     */
    BroadcastToBackward(const Tensor<T>& x, std::vector<int64_t> original_shape)
        : Node<T>(x), original_shape_(std::move(original_shape)) {}

    /**
     * Sum `propagated_grad` back onto `original_shape_` via `sum_to`.
     *
     * @param propagated_grad Upstream gradient in the broadcast shape.
     * @return `{dL/dx}`.
     *
     * @throws std::invalid_argument if `original_shape_` has a negative dimension.
     * @throws std::overflow_error if a numel product overflows.
     */
    std::vector<Tensor<T>> apply(const Tensor<T>& propagated_grad) override {
        return {sum_to(propagated_grad, original_shape_)};
    }
};

template <typename T>
class NarrowBackward : public Node<T> {
    int64_t dim_;
    int64_t start_;
    std::vector<int64_t> input_shape_;

public:
    /**
     * Construct the backward node for narrow.
     * Aliases `x` and stores the sliced dim, start index, and input shape.
     *
     * @param x Forward input (edge only; unused in apply).
     * @param dim Axis that was narrowed.
     * @param start First kept index along `dim`.
     */
    NarrowBackward(const Tensor<T>& x, int64_t dim, int64_t start)
        : Node<T>(x), dim_(dim), start_(start), input_shape_(x.shape()) {}

    /**
     * Scatter `propagated_grad` into a zero tensor of `input_shape_` at the
     * narrow window (`dim_`, `start_`).
     *
     * @param propagated_grad Upstream gradient dL/d(narrow output).
     * @return `{dL/dx}`.
     */
    std::vector<Tensor<T>> apply(const Tensor<T>& propagated_grad) override {
        std::vector<T> storage(static_cast<size_t>(
            Tensor<T>::compute_numel(input_shape_)), T{0});
        const auto g = propagated_grad.contiguous();
        const auto& gd = g.data();
        const int64_t rank = static_cast<int64_t>(input_shape_.size());
        const auto in_strides = Tensor<T>::compute_contiguous_strides(input_shape_);
        const auto g_shape = g.shape();
        const auto g_strides = Tensor<T>::compute_contiguous_strides(g_shape);
        for (int64_t f = 0; f < g.numel(); ++f) {
            int64_t rem = f;
            int64_t in_f = 0;
            for (int64_t d = 0; d < rank; ++d) {
                int64_t idx = rem / g_strides[static_cast<size_t>(d)];
                rem %= g_strides[static_cast<size_t>(d)];
                if (d == dim_)
                    idx += start_;
                in_f += idx * in_strides[static_cast<size_t>(d)];
            }
            storage[static_cast<size_t>(in_f)] = gd[static_cast<size_t>(f)];
        }
        return {Tensor<T>::from_operation_result(input_shape_, std::move(storage), false, nullptr)};
    }
};

template <typename T>
class CatBackward : public Node<T> {
    int64_t dim_;
    std::vector<int64_t> sizes_;
    std::vector<std::vector<int64_t>> in_shapes_;

public:
    /**
     * Construct the backward node for concatenate.
     * Aliases every input, records next edges, and stores per-input sizes
     * along `dim` plus full input shapes.
     *
     * @param inputs Forward tensors that were concatenated.
     * @param dim Axis of concatenation.
     */
    CatBackward(const std::vector<Tensor<T>>& inputs, int64_t dim)
        : dim_(dim) {
        this->saved_tensors.reserve(inputs.size());
        this->next_edges.reserve(inputs.size());
        sizes_.reserve(inputs.size());
        in_shapes_.reserve(inputs.size());
        for (const auto& t : inputs) {
            this->saved_tensors.emplace_back(Tensor<T>::alias(t));
            this->next_edges.emplace_back(Node<T>::get_next_edge(t));
            sizes_.push_back(t.shape()[static_cast<size_t>(dim)]);
            in_shapes_.push_back(t.shape());
        }
    }

    /**
     * Split `propagated_grad` along `dim_` into chunks of `sizes_[i]`.
     * Saved tensor values are unused; layout comes from `in_shapes_`.
     *
     * @param propagated_grad Upstream gradient dL/d(cat output).
     * @return Per-input gradients matching `inputs` order.
     */
    std::vector<Tensor<T>> apply(const Tensor<T>& propagated_grad) override {
        const auto g = propagated_grad.contiguous();
        const auto& gd = g.data();
        const auto g_shape = g.shape();
        const int64_t rank = g.rank();
        const auto g_strides = Tensor<T>::compute_contiguous_strides(g_shape);

        std::vector<Tensor<T>> grads;
        grads.reserve(sizes_.size());
        int64_t dim_offset = 0;
        for (size_t i = 0; i < sizes_.size(); ++i) {
            const auto& ishape = in_shapes_[i];
            const int64_t n = Tensor<T>::compute_numel(ishape);
            std::vector<T> storage(static_cast<size_t>(n));
            const auto in_strides = Tensor<T>::compute_contiguous_strides(ishape);
            for (int64_t f = 0; f < n; ++f) {
                int64_t remaining = f;
                int64_t g_f = 0;
                for (int64_t d = 0; d < rank; ++d) {
                    int64_t idx = remaining / in_strides[static_cast<size_t>(d)];
                    remaining %= in_strides[static_cast<size_t>(d)];
                    if (d == dim_)
                        idx += dim_offset;
                    g_f += idx * g_strides[static_cast<size_t>(d)];
                }
                storage[static_cast<size_t>(f)] = gd[static_cast<size_t>(g_f)];
            }
            grads.push_back(Tensor<T>::from_operation_result(
                ishape, std::move(storage), false, nullptr));
            dim_offset += sizes_[i];
        }
        return grads;
    }
};

template <typename OutT, typename InT>
class CastBackward : public Node<OutT> {
    Tensor<InT> input_;
    std::shared_ptr<Node<InT>> input_fn_;

public:
    /**
     * Construct the backward node for a dtype cast `InT` → `OutT`.
     * Aliases the source tensor and stores its `InT` grad_fn.
     *
     * @param x Forward input before the cast.
     */
    explicit CastBackward(const Tensor<InT>& x)
        : input_(Tensor<InT>::alias(x))
        , input_fn_(x.grad_fn()) {}

    /**
     * Recast `propagated_grad` to `InT` and continue the `InT` graph.
     * If `input_fn_` is set, calls `run_backward`; else if `input_` requires grad,
     * accumulates on the leaf. Returns an empty vector (no `OutT` next edges).
     *
     * @param propagated_grad Upstream gradient in `OutT`.
     * @return Empty vector.
     */
    std::vector<Tensor<OutT>> apply(const Tensor<OutT>& propagated_grad) override {
        const auto g = propagated_grad.contiguous();
        std::vector<InT> storage(static_cast<size_t>(g.numel()));
        const auto& gd = g.data();
        for (size_t i = 0; i < storage.size(); ++i)
            storage[i] = static_cast<InT>(gd[i]);
        const Tensor<InT> grad_in = Tensor<InT>::from_operation_result(
            g.shape(), std::move(storage), false, nullptr);

        if (input_fn_)
            run_backward(input_fn_, grad_in);
        else if (input_.requires_grad())
            input_.accumulate_grad(grad_in);
        return {};
    }
};

template <typename T>
class CastBackward<T, T> : public Node<T> {
public:
    /**
     * Construct the same-type cast backward node.
     * Aliases `x` for a normal `Node<T>` edge.
     *
     * @param x Forward input (same dtype as the output).
     */
    explicit CastBackward(const Tensor<T>& x) : Node<T>(x) {}

    /**
     * Identity Jacobian: return `propagated_grad` unchanged.
     *
     * @param propagated_grad Upstream gradient dL/d(cast output).
     * @return `{propagated_grad}`.
     */
    std::vector<Tensor<T>> apply(const Tensor<T>& propagated_grad) override {
        return {propagated_grad};
    }
};

} // namespace autograd
