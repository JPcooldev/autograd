#pragma once

#include <vector>
#include <memory>

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
    // we do not want to copy the input tensor, therefore Node<T>(), not Node<T>(x)
        : Node<T>(), dim0_(dim0), dim1_(dim1) 
    {
        this->next_edges.reserve(1);
        this->next_edges.emplace_back(Node<T>::get_next_edge(x));
    }

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
        : Node<T>(), original_shape_(std::move(original_shape))
    {
        this->next_edges.reserve(1);
        this->next_edges.emplace_back(Node<T>::get_next_edge(x));
    }

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
        : Node<T>(), original_shape_(std::move(original_shape))
    {
        this->next_edges.reserve(1);
        this->next_edges.emplace_back(Node<T>::get_next_edge(x));
    }

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
        : Node<T>(), original_shape_(std::move(original_shape))
    {
        this->next_edges.reserve(1);
        this->next_edges.emplace_back(Node<T>::get_next_edge(x));
    }

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
        : Node<T>(), original_shape_(std::move(original_shape))
    {
        this->next_edges.reserve(1);
        this->next_edges.emplace_back(Node<T>::get_next_edge(x));
    }

    std::vector<Tensor<T>> apply(const Tensor<T>& propagated_grad) override
    {
        return {propagated_grad.reshape(original_shape_)};
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
        : Node<T>(), original_shape_(std::move(original_shape))
    {
        this->next_edges.reserve(1);
        this->next_edges.emplace_back(Node<T>::get_next_edge(x));
    }

    std::vector<Tensor<T>> apply(const Tensor<T>& propagated_grad) override
    {
        // broadcast_to "virtually replicated" elements via stride-0 views.
        // The backward undoes that by summing all gradient contributions
        // that pointed at the same original element back together.
        return {sum_to(propagated_grad, original_shape_)};
    }
};

} // namespace autograd