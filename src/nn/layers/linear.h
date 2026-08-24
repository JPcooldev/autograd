#pragma once

#include <cmath>
#include <optional>
#include <stdexcept>
#include <vector>

#include "../../tensor/tensor.h"
#include "layer.h"
#include "../../ops/elementwise_ops.h"
#include "../../ops/linalg_ops.h"

namespace nn {

// ---------------------------------------------------------------------------
// Identity
// ---------------------------------------------------------------------------
// Passes the input through unchanged. Useful as a no-op placeholder inside
// module lists or for ablation experiments.

template <typename T>
class Identity : public Layer<T> {
public:
    tensor::Tensor<T> forward(const tensor::Tensor<T>& input) const {
        return input;
    }
    // std::vector<tensor::Tensor<T>*> parameters() override {
    //     return {nullptr};
    // }
};

// ---------------------------------------------------------------------------
// Linear
// ---------------------------------------------------------------------------
// Fully-connected layer:  y = x @ W.T + b   (or  y = x @ W.T  when bias=false)
//
// Parameters
//   weight  shape {out_features, in_features}  — kaiming_uniform(a=√5)
//                                                 = U(-1/√fan_in, 1/√fan_in)
//   bias    shape {out_features}               — U(-1/√fan_in, 1/√fan_in)
//                                                 present only when use_bias=true
//
// Forward input shapes:
//   1-D  {in_features}          → output {out_features}
//   2-D  {batch, in_features}   → output {batch, out_features}
//
// Saved-tensor behaviour
// ----------------------
// weight and bias are *leaf* tensors (requires_grad=true, no grad_fn).
// Node::snapshot_tensor detects this and stores a shared reference
// (Tensor::alias) instead of an O(N) deep copy — the same optimisation
// PyTorch applies via SavedVariable for leaf tensors.  The aliased tensors
// share the same data_ptr as the layer's parameters, so as long as the
// layer outlives the backward pass (the normal training-loop contract) the
// pointers remain valid.

template <typename T>
class Linear : public Layer<T> {
public:
    // {out_features, in_features}
    tensor::Tensor<T> weight;
    // {out_features} - nullopt when use_bias=false
    std::optional<tensor::Tensor<T>> bias;

    Linear(int64_t in_features, int64_t out_features, bool use_bias = true)
        : weight(tensor::Tensor<T>::kaiming_uniform(
              {out_features, in_features},
              std::sqrt(5.0),
              "fan_in",
              true)),
          in_features_(in_features),
          out_features_(out_features)
    {
        if (use_bias) {
            auto [fan_in, fan_out] = tensor::Tensor<T>::compute_fans(weight.shape());
            (void)fan_out;
            const T bound = static_cast<T>(
                1.0 / std::sqrt(static_cast<double>(fan_in)));
            bias = tensor::Tensor<T>::uniform({out_features}, -bound, bound, true);
        }
    }

    tensor::Tensor<T> forward(const tensor::Tensor<T>& input) const {
        if (input.rank() == 1) {
            if (input.shape()[0] != in_features_)
                throw std::invalid_argument(
                    "Linear: input size " + std::to_string(input.shape()[0]) +
                    " does not match in_features " + std::to_string(in_features_));

            // {in_features} → unsqueeze → {1, in_features} @ W.T → {1, out} → squeeze
            auto x2d = input.unsqueeze(0);                    // {1, in}
            auto out = ops::matmul(x2d, weight.transpose());  // {1, out}
            auto out1d = out.squeeze(0);                        // {out}
            if (bias) return ops::add(out1d, *bias);
            return out1d;

        } else if (input.rank() == 2) {
            if (input.shape()[1] != in_features_)
                throw std::invalid_argument(
                    "Linear: input column size " + std::to_string(input.shape()[1]) +
                    " does not match in_features " + std::to_string(in_features_));

            const int64_t batch = input.shape()[0];
            auto out = ops::matmul(input, weight.transpose());  // {batch, out}
            if (bias) {
                auto b = bias->unsqueeze(0).broadcast_to({batch, out_features_}).contiguous();
                return ops::add(out, b);
            }
            return out;

        } else {
            throw std::invalid_argument("Linear: input must be 1-D or 2-D");
        }
    }

    bool   has_bias()     const { return bias.has_value(); }
    int64_t in_features()  const { return in_features_; }
    int64_t out_features() const { return out_features_; }

    std::vector<tensor::Tensor<T>*> parameters() override {
        if (bias)
            return {&weight, &(*bias)};
        return {&weight};
    }

private:
    int64_t in_features_;
    int64_t out_features_;
};

} // namespace nn
