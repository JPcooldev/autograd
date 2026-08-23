#pragma once

#include <cmath>
#include <limits>
#include <vector>
#include <memory>

#include "../node.h"

namespace autograd {

using tensor::Tensor;

// ----- L1LossBackward -----
// Forward:  L = mean(|input - target|)
// Backward: dL/d(input)  =  sign(input - target) / N * g
//           dL/d(target) = -sign(input - target) / N * g
template <typename T>
class L1LossBackward : public Node<T> {
    std::vector<int64_t> shape_;
    int64_t numel_;
public:
    L1LossBackward(const Tensor<T>& input, const Tensor<T>& target)
        : Node<T>(input, target), shape_(input.shape()), numel_(input.numel()) {}

    std::vector<Tensor<T>> apply(const Tensor<T>& propagated_grad) override {
        const Tensor<T>& inp = this->saved_tensors[0];
        const Tensor<T>& tgt = this->saved_tensors[1];
        const T scale = propagated_grad.data()[0] / static_cast<T>(numel_);

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

// ----- L2LossBackward -----
// Forward:  L = sum((input - target)^2)   [un-normalised squared L2 error]
// Backward: dL/d(input)  =  2 * (input - target) * g
//           dL/d(target) = -2 * (input - target) * g
template <typename T>
class L2LossBackward : public Node<T> {
    std::vector<int64_t> shape_;
    int64_t numel_;
public:
    L2LossBackward(const Tensor<T>& input, const Tensor<T>& target)
        : Node<T>(input, target), shape_(input.shape()), numel_(input.numel()) {}

    std::vector<Tensor<T>> apply(const Tensor<T>& propagated_grad) override {
        const Tensor<T>& inp = this->saved_tensors[0];
        const Tensor<T>& tgt = this->saved_tensors[1];
        const T g = static_cast<T>(2) * propagated_grad.data()[0];

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

// ----- MSELossBackward -----
// Forward:  L = mean((input - target)^2)
// Backward: dL/d(input)  =  2 * (input - target) / N * g
//           dL/d(target) = -2 * (input - target) / N * g
template <typename T>
class MSELossBackward : public Node<T> {
    std::vector<int64_t> shape_;
    int64_t numel_;
public:
    MSELossBackward(const Tensor<T>& input, const Tensor<T>& target)
        : Node<T>(input, target), shape_(input.shape()), numel_(input.numel()) {}

    std::vector<Tensor<T>> apply(const Tensor<T>& propagated_grad) override {
        const Tensor<T>& inp = this->saved_tensors[0];
        const Tensor<T>& tgt = this->saved_tensors[1];
        const T scale = static_cast<T>(2) * propagated_grad.data()[0] / static_cast<T>(numel_);

        const size_t n = static_cast<size_t>(numel_);
        std::vector<T> gi(n), gt(n);
        for (size_t i = 0; i < n; ++i) {
            const T diff = inp.data()[i] - tgt.data()[i];
            gi[i] =  scale * diff;
            gt[i] = -scale * diff;
        }
        return {
            Tensor<T>::from_operation_result(shape_, std::move(gi), false, nullptr),
            Tensor<T>::from_operation_result(shape_, std::move(gt), false, nullptr)
        };
    }
};

// ----- NLLLossBackward -----
// Forward:  L = -(1/N_batch) * sum_{n,c} target_{n,c} * input_{n,c}
//           where input = log-probabilities, N_batch = numel / shape[dim]
// Backward: dL/d(input_{n,c})  = -target_{n,c} / N_batch * g
//           dL/d(target_{n,c}) = -input_{n,c}  / N_batch * g
template <typename T>
class NLLLossBackward : public Node<T> {
    std::vector<int64_t> shape_;
    int64_t numel_;
    int64_t n_batch_;
public:
    NLLLossBackward(const Tensor<T>& input, const Tensor<T>& target, int64_t n_batch)
        : Node<T>(input, target), shape_(input.shape()), numel_(input.numel()), n_batch_(n_batch) {}

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

// ----- CrossEntropyLossBackward -----
// Forward:  L = -(1/N_batch) * sum_{n,c} target_{n,c} * log_softmax(input)_{n,c}
//           (fused log-softmax + NLL for numerical stability)
// Backward: dL/d(input_{n,c})  = (softmax_{n,c} - target_{n,c}) / N_batch * g
//           dL/d(target_{n,c}) = -log_softmax(input)_{n,c} / N_batch * g
//           log_softmax is recovered as log(softmax_out) from the saved output.
template <typename T>
class CrossEntropyLossBackward : public Node<T> {
    Tensor<T> softmax_out_;  // saved no-grad softmax(input, dim)
    std::vector<int64_t> shape_;
    int64_t numel_;
    int64_t n_batch_;
public:
    CrossEntropyLossBackward(const Tensor<T>& input, const Tensor<T>& target,
                              const Tensor<T>& softmax_out, int64_t n_batch)
        : Node<T>(input, target),
          softmax_out_(Tensor<T>::alias(softmax_out)),
          shape_(input.shape()), numel_(input.numel()), n_batch_(n_batch) {}

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

// ----- BCELossBackward -----
// Forward:  L = -mean(target * log(input) + (1 - target) * log(1 - input))
//           input must lie in (0, 1); clamped to [eps, 1-eps] for numerical safety.
// Backward: dL/d(input)  = (input - target) / (input * (1 - input)) / N * g
//           dL/d(target) = -(log(input) - log(1 - input)) / N * g
template <typename T>
class BCELossBackward : public Node<T> {
    std::vector<int64_t> shape_;
    int64_t numel_;
public:
    BCELossBackward(const Tensor<T>& input, const Tensor<T>& target)
        : Node<T>(input, target), shape_(input.shape()), numel_(input.numel()) {}

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

// ----- BCEWithLogitsLossBackward -----
// Forward:  L = mean(log(1 + exp(input)) - input * target)  [numerically stable]
// Backward: dL/d(input)  = (sigmoid(input) - target) / N * g
//           dL/d(target) = -input / N * g
template <typename T>
class BCEWithLogitsLossBackward : public Node<T> {
    std::vector<int64_t> shape_;
    int64_t numel_;
public:
    BCEWithLogitsLossBackward(const Tensor<T>& input, const Tensor<T>& target)
        : Node<T>(input, target), shape_(input.shape()), numel_(input.numel()) {}

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

// ----- KLDivLossBackward -----
// Forward:  L = mean(target * (log(target) - input))   [input = log-probabilities]
// Backward: dL/d(input)  = -target / N * g
//           dL/d(target) = (log(target) - input + 1) / N * g
//           (d/dt [t*log(t)] = log(t) + 1; target clamped to avoid log(0))
template <typename T>
class KLDivLossBackward : public Node<T> {
    std::vector<int64_t> shape_;
    int64_t numel_;
public:
    KLDivLossBackward(const Tensor<T>& input, const Tensor<T>& target)
        : Node<T>(input, target), shape_(input.shape()), numel_(input.numel()) {}

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
