#pragma once

#include <memory>
#include <vector>

#include "node.h"

namespace autograd {

using tensor::Tensor;

// Terminal backward node for leaf tensors that require gradients.
//
// Every leaf tensor with requires_grad=true gets one AccumulateGrad node
// (created lazily via Tensor::ensure_accumulate_grad_fn()). All upstream
// backward nodes wire their next_edges to this node instead of storing nullptr.
//
// apply() is called by the engine exactly once per backward pass with the fully
// accumulated incoming gradient. It adds that gradient to accumulated_ so that
// multiple backward() calls without zero_grad() correctly sum contributions.
template <typename T>
class AccumulateGrad : public Node<T> {
    std::shared_ptr<Tensor<T>> accumulated_;

public:
    AccumulateGrad() : Node<T>() {}  // no saved_tensors, no next_edges

    std::vector<Tensor<T>> apply(const Tensor<T>& grad) override {
        if (!accumulated_) {
            // First contribution: copy the incoming gradient.
            accumulated_ = std::make_shared<Tensor<T>>(
                grad.shape(), grad.data(), false);
        } else {
            // Subsequent contributions: accumulate element-wise.
            // accumulated_ is always contiguous and freshly allocated.
            auto& dst = accumulated_->data();
            const auto& src = grad.data();
            for (size_t i = 0; i < dst.size(); ++i)
                dst[i] += src[i];
        }
        return {};  // leaf — no further edges to propagate to
    }

    // Returns the accumulated gradient, or nullptr if backward() was never run.
    const Tensor<T>* accumulated_grad() const {
        return accumulated_.get();
    }

    // Reset accumulated gradient (called by Tensor::zero_grad()).
    void zero() {
        accumulated_ = nullptr;
    }
};

} // namespace autograd
