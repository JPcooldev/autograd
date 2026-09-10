#pragma once

#include "../../ops/elementwise_ops.h"
#include "../../ops/loss_ops.h"
#include "../../tensor/tensor.h"
#include "layer.h"

namespace nn {

template <typename T>
class MSELoss : public Layer<T> {
public:
    /**
     * Compute mean squared error between `input` and `target`.
     * Calls `ops::l2_loss` with default mean reduction; returns a scalar tensor of shape `{}`.
     *
     * @param input Predictions; must have the same shape as `target`.
     * @param target Ground-truth values.
     * @return Scalar mean of `(input - target)^2`.
     *
     * @throws std::invalid_argument if `input` and `target` have different shapes.
     */
    tensor::Tensor<T> forward(const tensor::Tensor<T>& input, const tensor::Tensor<T>& target) const {
        return ops::l2_loss(input, target);
    }
};

template <typename T>
class L1Loss : public Layer<T> {
public:
    /**
     * Compute mean absolute error between `input` and `target`.
     * Calls `ops::l1_loss` with default mean reduction; returns a scalar tensor of shape `{}`.
     *
     * @param input Predictions; must have the same shape as `target`.
     * @param target Ground-truth values.
     * @return Scalar mean of `|input - target|`.
     *
     * @throws std::invalid_argument if `input` and `target` have different shapes.
     */
    tensor::Tensor<T> forward(const tensor::Tensor<T>& input, const tensor::Tensor<T>& target) const {
        return ops::l1_loss(input, target);
    }
};

template <typename T>
class CrossEntropyLoss : public Layer<T> {
public:
    /**
     * Compute fused log-softmax + NLL with soft labels.
     * Calls `ops::cross_entropy_loss` (class axis `dim = -1`); `target` must be probabilities
     * with the same shape as logits. Returns a scalar tensor of shape `{}`.
     *
     * @param input Logits.
     * @param target Soft labels, same shape as `input`.
     * @return Scalar mean NLL of log-softmax(`input`) against `target`.
     *
     * @throws std::invalid_argument if `input` and `target` have different shapes.
     * @throws std::invalid_argument if `input` is a scalar.
     */
    tensor::Tensor<T> forward(const tensor::Tensor<T>& input, const tensor::Tensor<T>& target) const {
        return ops::cross_entropy_loss(input, target);
    }
};

template <typename T>
class BCELoss : public Layer<T> {
public:
    /**
     * Compute binary cross-entropy from probabilities.
     * Calls `ops::bce_loss`; input is clamped to `[eps, 1-eps]` inside the op. Returns a scalar `{}`.
     *
     * @param input Probabilities in (0, 1), same shape as `target`.
     * @param target Binary (or soft) targets.
     * @return Scalar mean BCE.
     *
     * @throws std::invalid_argument if `input` and `target` have different shapes.
     */
    tensor::Tensor<T> forward(const tensor::Tensor<T>& input, const tensor::Tensor<T>& target) const {
        return ops::bce_loss(input, target);
    }
};

template <typename T>
class BCEWithLogitsLoss : public Layer<T> {
public:
    /**
     * Compute numerically stable sigmoid + BCE from logits.
     * Calls `ops::bce_with_logits_loss`. Returns a scalar tensor of shape `{}`.
     *
     * @param input Logits, same shape as `target`.
     * @param target Binary (or soft) targets.
     * @return Scalar mean fused BCE-with-logits.
     *
     * @throws std::invalid_argument if `input` and `target` have different shapes.
     */
    tensor::Tensor<T> forward(const tensor::Tensor<T>& input, const tensor::Tensor<T>& target) const {
        return ops::bce_with_logits_loss(input, target);
    }
};

template <typename T>
class KLDivLoss : public Layer<T> {
public:
    /**
     * Compute mean KL divergence with log-probability inputs.
     * Calls `ops::kl_div_loss`: `L = mean(target * (log(target) - input))` with `target` clamped
     * away from 0 for `log`. Returns a scalar tensor of shape `{}`.
     *
     * @param input Log-probabilities, same shape as `target`.
     * @param target Probabilities.
     * @return Scalar mean KL.
     *
     * @throws std::invalid_argument if `input` and `target` have different shapes.
     */
    tensor::Tensor<T> forward(const tensor::Tensor<T>& input, const tensor::Tensor<T>& target) const {
        return ops::kl_div_loss(input, target);
    }
};

} // namespace nn
