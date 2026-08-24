#pragma once

#include <memory>
#include <vector>

#include "../engine.h"
#include "../node.h"

namespace autograd {

using tensor::Tensor;

template <typename T>
class TransposeBackward : public Node<T> {
    private: // for inverse transpose
    int64_t dim0_;
    int64_t dim1_;
public:
    TransposeBackward(const Tensor<T>& x, const int64_t dim0, const int64_t dim1)
        : Node<T>(x), dim0_(dim0), dim1_(dim1)
    {}

    std::vector<Tensor<T>> apply(const Tensor<T>& propagated_grad) override 
    {
        // Inverse of a transpose is the same transpose over the same dimensions.
        return {propagated_grad.transpose(dim0_, dim1_)};
    }
};

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
    ReshapeBackward(const Tensor<T>& x, std::vector<int64_t> original_shape)
        : Node<T>(x), original_shape_(std::move(original_shape))
    {}

    std::vector<Tensor<T>> apply(const Tensor<T>& propagated_grad) override
    {
        return {propagated_grad.reshape(original_shape_)};
    }
};

// Reduce `grad` (shape: out_shape) back to `target_shape` by summing over all
// axes that were introduced by broadcasting:
//   - prepended axes (rank_diff leading dims not present in the original tensor)
//   - axes where target_shape[i] == 1 but out_shape[i] > 1
//
// This is the mathematical inverse of broadcast_to: broadcasting repeats an
// element N times along an axis, so the gradient contribution from all N
// positions must be summed back into the single original element.
//
// Assumption: `grad` is contiguous (flat data index == logical index).
// This holds because all backward-pass outputs are freshly allocated tensors.
template <typename T>
Tensor<T> sum_to(const Tensor<T>& grad, const std::vector<int64_t>& target_shape) {
    // Fast path: no reduction required.
    if (grad.shape() == target_shape) {
        return Tensor<T>::from_operation_result(
            target_shape,
            std::vector<T>(grad.data().begin(), grad.data().end()),
            false, nullptr
        );
    }

    const int64_t grad_rank   = grad.rank();
    const int64_t target_rank = static_cast<int64_t>(target_shape.size());
    // broadcast_to always keeps or increases rank, so grad_rank >= target_rank.
    const int64_t rank_diff   = grad_rank - target_rank;

    // Contiguous strides for grad: used to decompose a flat index into a
    // per-axis multi-index without needing a separate modulo-only loop.
    const auto grad_cont_strides = Tensor<T>::compute_contiguous_strides(grad.shape());
    // Contiguous strides for the output: used to re-compose the reduced multi-index.
    const auto out_cont_strides  = Tensor<T>::compute_contiguous_strides(target_shape);

    const int64_t out_numel = Tensor<T>::compute_numel(target_shape);
    std::vector<T> out(static_cast<size_t>(out_numel), T{0});

    for (int64_t flat = 0; flat < grad.numel(); ++flat) {
        int64_t out_flat  = 0;
        int64_t remaining = flat;

        for (int64_t d = 0; d < grad_rank; ++d) {
            // Decompose remaining flat offset into the index along dim d.
            const int64_t idx = remaining / grad_cont_strides[static_cast<size_t>(d)];
            remaining        %= grad_cont_strides[static_cast<size_t>(d)];

            // The corresponding dimension in the target (negative → prepended dim).
            const int64_t td = d - rank_diff;

            // Prepended dims and size-1 target dims are broadcast axes.
            // Their index does not contribute to the output position —
            // all positions along a broadcast axis map to the same output element.
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
    FlattenBackward(const Tensor<T>& x, std::vector<int64_t> original_shape)
        : Node<T>(x), original_shape_(std::move(original_shape))
    {}

    std::vector<Tensor<T>> apply(const Tensor<T>& propagated_grad) override
    {
        return {propagated_grad.reshape(original_shape_)};
    }
};

// Backward for squeeze (both the no-arg and dim variant): reshapes grad back
// to the pre-squeeze shape, which is the exact inverse of squeeze.
template <typename T>
class SqueezeBackward : public Node<T> {
    std::vector<int64_t> original_shape_;
public:
    SqueezeBackward(const Tensor<T>& x, std::vector<int64_t> original_shape)
        : Node<T>(x), original_shape_(std::move(original_shape))
    {}

    std::vector<Tensor<T>> apply(const Tensor<T>& propagated_grad) override
    {
        return {propagated_grad.reshape(original_shape_)};
    }
};

// Backward for unsqueeze: reshapes grad back to the pre-unsqueeze shape,
// which is the exact inverse of unsqueeze.
template <typename T>
class UnsqueezeBackward : public Node<T> {
    std::vector<int64_t> original_shape_;
public:
    UnsqueezeBackward(const Tensor<T>& x, std::vector<int64_t> original_shape)
        : Node<T>(x), original_shape_(std::move(original_shape))
    {}

    std::vector<Tensor<T>> apply(const Tensor<T>& propagated_grad) override
    {
        return {propagated_grad.reshape(original_shape_)};
    }
};

// Layout-only op: values and logical shape are unchanged. The incoming
// gradient is already in logical (row-major) order, so backward is identity.
template <typename T>
class ContiguousBackward : public Node<T> {
public:
    explicit ContiguousBackward(const Tensor<T>& x) : Node<T>(x) {}

    std::vector<Tensor<T>> apply(const Tensor<T>& propagated_grad) override
    {
        return {propagated_grad};
    }
};

template <typename T>
class BroadcastToBackward : public Node<T> {
    // We only need the original shape for the backward sum reduction.
    // The original tensor data is not required, so we do not save it
    // (same pattern as TransposeBackward / ReshapeBackward).
    std::vector<int64_t> original_shape_;
public:
    BroadcastToBackward(const Tensor<T>& x, std::vector<int64_t> original_shape)
        : Node<T>(x), original_shape_(std::move(original_shape))
    {}

    std::vector<Tensor<T>> apply(const Tensor<T>& propagated_grad) override
    {
        // broadcast_to "virtually replicated" elements via stride-0 views.
        // The backward undoes that by summing all gradient contributions
        // that pointed at the same original element back together.
        return {sum_to(propagated_grad, original_shape_)};
    }
};

template <typename T>
class NarrowBackward : public Node<T> {
    int64_t dim_;
    int64_t start_;
    std::vector<int64_t> input_shape_;
public:
    NarrowBackward(const Tensor<T>& x, int64_t dim, int64_t start)
        : Node<T>(x), dim_(dim), start_(start), input_shape_(x.shape())
    {}

    std::vector<Tensor<T>> apply(const Tensor<T>& propagated_grad) override
    {
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
                if (d == dim_) idx += start_;
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
    CatBackward(const std::vector<Tensor<T>>& inputs, int64_t dim)
        : dim_(dim)
    {
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

    std::vector<Tensor<T>> apply(const Tensor<T>& propagated_grad) override
    {
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
                    if (d == dim_) idx += dim_offset;
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

// Backward of y = cast_{InT → OutT}(x): recast the incoming gradient to InT.
// The Jacobian is the identity, so apply() is a dtype conversion of `propagated_grad`.
//
// Same-type (`CastBackward<T, T>`): a normal Node<T> edge; the engine continues.
// Cross-type: Node<OutT> cannot hold Node<InT> edges, so apply() either
// accumulate_grad()s into a leaf or run_backward()s the InT subgraph.

template <typename OutT, typename InT>
class CastBackward : public Node<OutT> {
    Tensor<InT> input_;
    std::shared_ptr<Node<InT>> input_fn_;
public:
    explicit CastBackward(const Tensor<InT>& x)
        : input_(Tensor<InT>::alias(x))
        , input_fn_(x.grad_fn())
    {}

    std::vector<Tensor<OutT>> apply(const Tensor<OutT>& propagated_grad) override
    {
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
    explicit CastBackward(const Tensor<T>& x) : Node<T>(x) {}

    std::vector<Tensor<T>> apply(const Tensor<T>& propagated_grad) override
    {
        return {propagated_grad};
    }
};

} // namespace autograd