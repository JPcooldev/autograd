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
    /**
     * Return the input tensor unchanged.
     * No copy is made; the same tensor object is returned.
     *
     * @param input The tensor to pass through.
     * @return `input`.
     */
    tensor::Tensor<T> forward(const tensor::Tensor<T>& input) const {
        return input;
    }
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
//   1-D  {in_features}                → output {out_features}
//   2-D  {batch, in_features}         → output {batch, out_features}
//   N-D  {..., in_features}           → output {..., out_features}
//        (last axis is the feature dim; leading axes are flattened for matmul)
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
private:
    int64_t in_features_;
    int64_t out_features_;

public:
    // {out_features, in_features}
    tensor::Tensor<T> weight;
    // {out_features} - nullopt when use_bias=false
    std::optional<tensor::Tensor<T>> bias;

    /**
     * Construct a fully-connected layer.
     * Initializes `weight` with kaiming_uniform(a=√5, fan_in). If `use_bias`,
     * initializes `bias` with U(-1/√fan_in, 1/√fan_in) from `compute_fans(weight)`.
     *
     * @param in_features Size of each input sample.
     * @param out_features Size of each output sample.
     * @param use_bias If true, allocate and initialize a bias of shape `{out_features}`.
     */
    Linear(int64_t in_features, int64_t out_features, bool use_bias = true)
        : in_features_(in_features),
          out_features_(out_features),
          weight(tensor::Tensor<T>::kaiming_uniform({out_features, in_features}, std::sqrt(5.0), "fan_in", true)) {
        if (use_bias) {
            auto [fan_in, fan_out] = tensor::Tensor<T>::compute_fans(weight.shape());
            (void)fan_out;
            const T bound = static_cast<T>(1.0 / std::sqrt(static_cast<double>(fan_in)));
            bias = tensor::Tensor<T>::uniform({out_features}, -bound, bound, true);
        }
    }

    /**
     * Apply the affine transform `y = x @ W.T [+ b]` on the last axis.
     * Rank-1 inputs are unsqueezed to a batch of 1, multiplied by `weight.transpose()`,
     * then squeezed. Rank-2 inputs are multiplied directly. Rank ≥ 3 flattens every
     * leading axis, multiplies, then reshapes back; bias is broadcast over the batch.
     *
     * @param input Rank-1 `{in_features}` or `{..., in_features}`.
     * @return Rank-1 `{out_features}` or `{..., out_features}`.
     *
     * @throws std::invalid_argument if `input` is a scalar.
     * @throws std::invalid_argument if the last dimension of `input` does not equal `in_features`.
     */
    tensor::Tensor<T> forward(const tensor::Tensor<T>& input) const {
        if (input.rank() < 1)
            throw std::invalid_argument("Linear: input must have rank >= 1");
        if (input.shape().back() != in_features_)
            throw std::invalid_argument(
                "Linear: last dim " + std::to_string(input.shape().back()) +
                " does not match in_features " + std::to_string(in_features_));

        if (input.rank() == 1) {
            auto x2d = input.unsqueeze(0);
            auto out = ops::matmul(x2d, weight.transpose());
            auto out1d = out.squeeze(0);
            if (bias)
                return ops::add(out1d, *bias);
            return out1d;
        }

        std::vector<int64_t> out_shape = input.shape();
        out_shape.back() = out_features_;
        auto x2d = input.rank() == 2 ? input : input.flatten(0, -2);
        auto out = ops::matmul(x2d, weight.transpose());
        if (bias) {
            auto b = bias->unsqueeze(0).broadcast_to(out.shape()).contiguous();
            out = ops::add(out, b);
        }
        if (input.rank() == 2)
            return out;
        return out.reshape(out_shape);
    }

    /**
     * Report whether this layer has a bias tensor.
     *
     * @return True if `bias` holds a tensor.
     */
    bool has_bias() const { return bias.has_value(); }

    /**
     * Return the configured input feature count.
     *
     * @return `in_features_`.
     */
    int64_t in_features() const { return in_features_; }

    /**
     * Return the configured output feature count.
     *
     * @return `out_features_`.
     */
    int64_t out_features() const { return out_features_; }

    /**
     * Return named learnable parameters.
     * Always includes `weight`; appends `bias` when present.
     *
     * @return `{{"weight", &weight}, {"bias", &*bias}}` or just weight.
     */
    NamedTensorList<T> named_parameters() override {
        return detail::named_weight_bias(weight, bias);
    }
};

} // namespace nn
