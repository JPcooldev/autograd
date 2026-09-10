#pragma once

#include <cmath>
#include <limits>
#include <vector>
#include <memory>

#include "../node.h"

namespace autograd {

using tensor::Tensor;

template <typename T>
class L1LossBackward : public Node<T> {
    std::vector<int64_t> shape_;
    int64_t numel_;
    bool reduction_;

public:
    /**
     * Construct the backward node for L1 loss.
     * Aliases `input` and `target` and stores shape, numel, and reduction.
     *
     * @param input Forward predictions.
     * @param target Forward targets.
     * @param reduction True if the forward loss was a mean, false if a sum.
     */
    L1LossBackward(const Tensor<T>& input, const Tensor<T>& target, bool reduction)
        : Node<T>(input, target), shape_(input.shape()), numel_(input.numel()), reduction_(reduction) {}

    /**
     * Compute L1 loss gradients.
     * Uses saved input and target: dL/d(input) = sign(input - target) / denom * g
     * and dL/d(target) = -that, with denom = N if `reduction_` else 1.
     *
     * @param propagated_grad Scalar upstream gradient dL/dL_loss.
     * @return `{dL/d(input), dL/d(target)}`.
     */
    std::vector<Tensor<T>> apply(const Tensor<T>& propagated_grad) override {
        const Tensor<T>& inp = this->saved_tensors[0];
        const Tensor<T>& tgt = this->saved_tensors[1];
        const T denom = reduction_ ? static_cast<T>(numel_) : T{1};
        const T scale = propagated_grad.data()[0] / denom;

        const size_t n = static_cast<size_t>(numel_);
        std::vector<T> gi(n), gt(n);
        for (size_t i = 0; i < n; ++i) {
            const T diff = inp.data()[i] - tgt.data()[i];
            const T sign = diff > T{0} ? T{1} : (diff < T{0} ? T{-1} : T{0});
            gi[i] =  sign * scale;
            gt[i] = -sign * scale;
        }
        return {
            Tensor<T>::from_operation_result(shape_, std::move(gi), false, nullptr),
            Tensor<T>::from_operation_result(shape_, std::move(gt), false, nullptr)
        };
    }
};

template <typename T>
class L2LossBackward : public Node<T> {
    std::vector<int64_t> shape_;
    int64_t numel_;
    bool reduction_;

public:
    /**
     * Construct the backward node for L2 (squared-error) loss.
     * Aliases `input` and `target` and stores shape, numel, and reduction.
     *
     * @param input Forward predictions.
     * @param target Forward targets.
     * @param reduction True if the forward loss was a mean, false if a sum.
     */
    L2LossBackward(const Tensor<T>& input, const Tensor<T>& target, bool reduction)
        : Node<T>(input, target), shape_(input.shape()), numel_(input.numel()), reduction_(reduction) {}

    /**
     * Compute L2 loss gradients.
     * Uses saved input and target: dL/d(input) = 2 * (input - target) / denom * g
     * and dL/d(target) = -that, with denom = N if `reduction_` else 1.
     *
     * @param propagated_grad Scalar upstream gradient dL/dL_loss.
     * @return `{dL/d(input), dL/d(target)}`.
     */
    std::vector<Tensor<T>> apply(const Tensor<T>& propagated_grad) override {
        const Tensor<T>& inp = this->saved_tensors[0];
        const Tensor<T>& tgt = this->saved_tensors[1];
        const T denom = reduction_ ? static_cast<T>(numel_) : T{1};
        const T g = static_cast<T>(2) * propagated_grad.data()[0] / denom;

        const size_t n = static_cast<size_t>(numel_);
        std::vector<T> gi(n), gt(n);
        for (size_t i = 0; i < n; ++i) {
            const T diff = inp.data()[i] - tgt.data()[i];
            gi[i] =  g * diff;
            gt[i] = -g * diff;
        }
        return {
            Tensor<T>::from_operation_result(shape_, std::move(gi), false, nullptr),
            Tensor<T>::from_operation_result(shape_, std::move(gt), false, nullptr)
        };
    }
};

template <typename T>
class NLLLossBackward : public Node<T> {
    std::vector<int64_t> shape_;
    int64_t numel_;
    int64_t n_batch_;

public:
    /**
     * Construct the backward node for negative log-likelihood loss.
     * Aliases `input` (log-probs) and `target` and stores shape, numel, and batch size.
     *
     * @param input Forward log-probabilities.
     * @param target Forward class weights / one-hot targets.
     * @param n_batch Batch size used as the forward mean denominator.
     */
    NLLLossBackward(const Tensor<T>& input, const Tensor<T>& target, int64_t n_batch)
        : Node<T>(input, target), shape_(input.shape()), numel_(input.numel()), n_batch_(n_batch) {}

    /**
     * Compute NLL gradients: dL/d(input) = -target / n_batch * g,
     * dL/d(target) = -input / n_batch * g.
     * Uses saved input and target.
     *
     * @param propagated_grad Scalar upstream gradient dL/dL_loss.
     * @return `{dL/d(input), dL/d(target)}`.
     */
    std::vector<Tensor<T>> apply(const Tensor<T>& propagated_grad) override {
        const Tensor<T>& inp = this->saved_tensors[0];
        const Tensor<T>& tgt = this->saved_tensors[1];
        const T scale = propagated_grad.data()[0] / static_cast<T>(n_batch_);

        const size_t n = static_cast<size_t>(numel_);
        std::vector<T> gi(n), gt(n);
        for (size_t i = 0; i < n; ++i) {
            gi[i] = -tgt.data()[i] * scale;
            gt[i] = -inp.data()[i] * scale;
        }
        return {
            Tensor<T>::from_operation_result(shape_, std::move(gi), false, nullptr),
            Tensor<T>::from_operation_result(shape_, std::move(gt), false, nullptr)
        };
    }
};

template <typename T>
class CrossEntropyLossBackward : public Node<T> {
    Tensor<T> softmax_out_;  // saved no-grad softmax(input, dim)
    std::vector<int64_t> shape_;
    int64_t numel_;
    int64_t n_batch_;

public:
    /**
     * Construct the backward node for fused softmax + NLL (cross entropy).
     * Aliases `input` and `target`, aliases `softmax_out` into `softmax_out_`,
     * and stores shape, numel, and batch size.
     *
     * @param input Forward logits (saved for the graph edge; unused in apply).
     * @param target Forward targets.
     * @param softmax_out Detached softmax(input) from the forward pass.
     * @param n_batch Batch size used as the forward mean denominator.
     */
    CrossEntropyLossBackward(
        const Tensor<T>& input,
        const Tensor<T>& target,
        const Tensor<T>& softmax_out,
        int64_t n_batch
    )
        : Node<T>(input, target),
          softmax_out_(Tensor<T>::alias(softmax_out)),
          shape_(input.shape()),
          numel_(input.numel()),
          n_batch_(n_batch) {}

    /**
     * Compute cross-entropy gradients from saved softmax and target.
     * dL/d(input) = (softmax - target) / n_batch * g;
     * dL/d(target) = -log(softmax) / n_batch * g with softmax clamped by epsilon.
     * Saved input is unused.
     *
     * @param propagated_grad Scalar upstream gradient dL/dL_loss.
     * @return `{dL/d(input), dL/d(target)}`.
     */
    std::vector<Tensor<T>> apply(const Tensor<T>& propagated_grad) override {
        const Tensor<T>& tgt = this->saved_tensors[1];
        const T scale = propagated_grad.data()[0] / static_cast<T>(n_batch_);
        constexpr T eps = std::numeric_limits<T>::epsilon();

        const size_t n = static_cast<size_t>(numel_);
        std::vector<T> gi(n), gt(n);
        for (size_t i = 0; i < n; ++i) {
            const T s = softmax_out_.data()[i];
            gi[i] = (s - tgt.data()[i]) * scale;
            gt[i] = -static_cast<T>(std::log(std::max(s, eps))) * scale;
        }
        return {
            Tensor<T>::from_operation_result(shape_, std::move(gi), false, nullptr),
            Tensor<T>::from_operation_result(shape_, std::move(gt), false, nullptr)
        };
    }
};

template <typename T>
class BCELossBackward : public Node<T> {
    std::vector<int64_t> shape_;
    int64_t numel_;

public:
    /**
     * Construct the backward node for binary cross-entropy.
     * Aliases `input` and `target` and stores shape and numel.
     *
     * @param input Forward probabilities in (0, 1).
     * @param target Forward binary targets.
     */
    BCELossBackward(const Tensor<T>& input, const Tensor<T>& target)
        : Node<T>(input, target), shape_(input.shape()), numel_(input.numel()) {}

    /**
     * Compute BCE gradients from saved input and target.
     * Input is clamped to [eps, 1-eps]; dL/d(input) = (p - t) / (p*(1-p)) / N * g
     * and dL/d(target) = -(log(p) - log(1-p)) / N * g.
     *
     * @param propagated_grad Scalar upstream gradient dL/dL_loss.
     * @return `{dL/d(input), dL/d(target)}`.
     */
    std::vector<Tensor<T>> apply(const Tensor<T>& propagated_grad) override {
        const Tensor<T>& inp = this->saved_tensors[0];
        const Tensor<T>& tgt = this->saved_tensors[1];
        const T scale = propagated_grad.data()[0] / static_cast<T>(numel_);
        constexpr T eps = std::numeric_limits<T>::epsilon();

        const size_t n = static_cast<size_t>(numel_);
        std::vector<T> gi(n), gt(n);
        for (size_t i = 0; i < n; ++i) {
            const T p  = std::max(std::min(inp.data()[i], static_cast<T>(1) - eps), eps);
            const T t  = tgt.data()[i];
            const T p1 = static_cast<T>(1) - p;
            gi[i] = (p - t) / (p * p1) * scale;
            gt[i] = -(static_cast<T>(std::log(p)) - static_cast<T>(std::log(p1))) * scale;
        }
        return {
            Tensor<T>::from_operation_result(shape_, std::move(gi), false, nullptr),
            Tensor<T>::from_operation_result(shape_, std::move(gt), false, nullptr)
        };
    }
};

template <typename T>
class BCEWithLogitsLossBackward : public Node<T> {
    std::vector<int64_t> shape_;
    int64_t numel_;

public:
    /**
     * Construct the backward node for BCE-with-logits.
     * Aliases `input` and `target` and stores shape and numel.
     *
     * @param input Forward logits.
     * @param target Forward binary targets.
     */
    BCEWithLogitsLossBackward(const Tensor<T>& input, const Tensor<T>& target)
        : Node<T>(input, target), shape_(input.shape()), numel_(input.numel()) {}

    /**
     * Compute BCE-with-logits gradients from saved input and target.
     * dL/d(input) = (sigmoid(input) - target) / N * g,
     * dL/d(target) = -input / N * g.
     *
     * @param propagated_grad Scalar upstream gradient dL/dL_loss.
     * @return `{dL/d(input), dL/d(target)}`.
     */
    std::vector<Tensor<T>> apply(const Tensor<T>& propagated_grad) override {
        const Tensor<T>& inp = this->saved_tensors[0];
        const Tensor<T>& tgt = this->saved_tensors[1];
        const T scale = propagated_grad.data()[0] / static_cast<T>(numel_);

        const size_t n = static_cast<size_t>(numel_);
        std::vector<T> gi(n), gt(n);
        for (size_t i = 0; i < n; ++i) {
            const T x   = inp.data()[i];
            const T sig = static_cast<T>(1) / (static_cast<T>(1) + static_cast<T>(std::exp(-x)));
            gi[i] = (sig - tgt.data()[i]) * scale;
            gt[i] = -x * scale;
        }
        return {
            Tensor<T>::from_operation_result(shape_, std::move(gi), false, nullptr),
            Tensor<T>::from_operation_result(shape_, std::move(gt), false, nullptr)
        };
    }
};

template <typename T>
class KLDivLossBackward : public Node<T> {
    std::vector<int64_t> shape_;
    int64_t numel_;

public:
    /**
     * Construct the backward node for KL divergence (mean reduction).
     * Aliases `input` (log-probs) and `target` and stores shape and numel.
     *
     * @param input Forward log-probabilities.
     * @param target Forward probabilities.
     */
    KLDivLossBackward(const Tensor<T>& input, const Tensor<T>& target)
        : Node<T>(input, target), shape_(input.shape()), numel_(input.numel()) {}

    /**
     * Compute KL-div gradients from saved input and target.
     * dL/d(input) = -target / N * g;
     * dL/d(target) = (log(target) - input + 1) / N * g with target clamped for log.
     *
     * @param propagated_grad Scalar upstream gradient dL/dL_loss.
     * @return `{dL/d(input), dL/d(target)}`.
     */
    std::vector<Tensor<T>> apply(const Tensor<T>& propagated_grad) override {
        const Tensor<T>& inp = this->saved_tensors[0];
        const Tensor<T>& tgt = this->saved_tensors[1];
        const T scale = propagated_grad.data()[0] / static_cast<T>(numel_);
        constexpr T eps = std::numeric_limits<T>::epsilon();

        const size_t n = static_cast<size_t>(numel_);
        std::vector<T> gi(n), gt(n);
        for (size_t i = 0; i < n; ++i) {
            const T t     = tgt.data()[i];
            const T log_t = static_cast<T>(std::log(std::max(t, eps)));
            gi[i] = -t * scale;
            gt[i] = (log_t - inp.data()[i] + static_cast<T>(1)) * scale;
        }
        return {
            Tensor<T>::from_operation_result(shape_, std::move(gi), false, nullptr),
            Tensor<T>::from_operation_result(shape_, std::move(gt), false, nullptr)
        };
    }
};

} // namespace autograd
