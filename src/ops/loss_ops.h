#pragma once

#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

#include "../tensor/tensor.h"
#include "../autograd/grad_context.h"
#include "../autograd/backward_ops/backward_loss_ops.h"

namespace ops {

// ----- l1_loss -----
// Mean absolute error: L = mean(|input - target|)
// Returns a scalar tensor of shape {}.
template <typename T>
tensor::Tensor<T> l1_loss(const tensor::Tensor<T>& input, const tensor::Tensor<T>& target)
{
    check_same_shapes(input, target, "l1_loss");

    T sum = T{0};
    for (int64_t i = 0; i < input.numel(); ++i)
        sum += static_cast<T>(std::abs(input.data()[static_cast<size_t>(i)]
                                      - target.data()[static_cast<size_t>(i)]));
    const T loss = sum / static_cast<T>(input.numel());

    const bool requires_grad = autograd::is_grad_enabled()
                             && (input.requires_grad() || target.requires_grad());
    std::shared_ptr<autograd::Node<T>> grad_fn = nullptr;
    if (requires_grad)
        grad_fn = std::make_shared<autograd::L1LossBackward<T>>(input, target);

    return tensor::Tensor<T>::from_operation_result(
        {}, std::vector<T>{loss}, requires_grad, std::move(grad_fn));
}

// ----- l2_loss -----
// Un-normalised squared L2 error: L = sum((input - target)^2)
// Returns a scalar tensor of shape {}.
template <typename T>
tensor::Tensor<T> l2_loss(const tensor::Tensor<T>& input, const tensor::Tensor<T>& target)
{
    check_same_shapes(input, target, "l2_loss");

    T sum = T{0};
    for (int64_t i = 0; i < input.numel(); ++i) {
        const T diff = input.data()[static_cast<size_t>(i)]
                     - target.data()[static_cast<size_t>(i)];
        sum += diff * diff;
    }

    const bool requires_grad = autograd::is_grad_enabled()
                             && (input.requires_grad() || target.requires_grad());
    std::shared_ptr<autograd::Node<T>> grad_fn = nullptr;
    if (requires_grad)
        grad_fn = std::make_shared<autograd::L2LossBackward<T>>(input, target);

    return tensor::Tensor<T>::from_operation_result(
        {}, std::vector<T>{sum}, requires_grad, std::move(grad_fn));
}

// ----- mse_loss -----
// Mean squared error: L = mean((input - target)^2) = l2_loss / N
// Returns a scalar tensor of shape {}.
template <typename T>
tensor::Tensor<T> mse_loss(const tensor::Tensor<T>& input, const tensor::Tensor<T>& target)
{
    check_same_shapes(input, target, "mse_loss");

    T sum = T{0};
    for (int64_t i = 0; i < input.numel(); ++i) {
        const T diff = input.data()[static_cast<size_t>(i)]
                     - target.data()[static_cast<size_t>(i)];
        sum += diff * diff;
    }
    const T loss = sum / static_cast<T>(input.numel());

    const bool requires_grad = autograd::is_grad_enabled()
                             && (input.requires_grad() || target.requires_grad());
    std::shared_ptr<autograd::Node<T>> grad_fn = nullptr;
    if (requires_grad)
        grad_fn = std::make_shared<autograd::MSELossBackward<T>>(input, target);

    return tensor::Tensor<T>::from_operation_result(
        {}, std::vector<T>{loss}, requires_grad, std::move(grad_fn));
}

// ----- nll_loss -----
// Negative log-likelihood loss (input must be log-probabilities):
//   L = -(1/N_batch) * sum_{n,c} target_{n,c} * input_{n,c}
// where N_batch = numel / shape[dim] (number of samples; dim is the class axis).
// Returns a scalar tensor of shape {}.
template <typename T>
tensor::Tensor<T> nll_loss(const tensor::Tensor<T>& input, const tensor::Tensor<T>& target,
                            int64_t dim = -1)
{
    check_same_shapes(input, target, "nll_loss");
    if (input.numel() == 0)
        throw std::invalid_argument("nll_loss requires a non-empty tensor");

    dim = tensor::Tensor<T>::normalize_dimension(dim, input.rank());
    const int64_t n_batch = input.numel() / input.shape()[static_cast<size_t>(dim)];

    T sum = T{0};
    for (int64_t i = 0; i < input.numel(); ++i)
        sum -= target.data()[static_cast<size_t>(i)] * input.data()[static_cast<size_t>(i)];
    const T loss = sum / static_cast<T>(n_batch);

    const bool requires_grad = autograd::is_grad_enabled()
                             && (input.requires_grad() || target.requires_grad());
    std::shared_ptr<autograd::Node<T>> grad_fn = nullptr;
    if (requires_grad)
        grad_fn = std::make_shared<autograd::NLLLossBackward<T>>(input, target, n_batch);

    return tensor::Tensor<T>::from_operation_result(
        {}, std::vector<T>{loss}, requires_grad, std::move(grad_fn));
}

// ----- cross_entropy_loss -----
// Fused log-softmax + NLL (numerically stable):
//   L = -(1/N_batch) * sum_{n,c} target_{n,c} * log_softmax(input, dim)_{n,c}
// where N_batch = numel / shape[dim] (number of samples; dim is the class axis).
// target must be soft labels (probabilities) with the same shape as input.
// Returns a scalar tensor of shape {}.
template <typename T>
tensor::Tensor<T> cross_entropy_loss(const tensor::Tensor<T>& input,
                                      const tensor::Tensor<T>& target,
                                      int64_t dim = -1)
{
    check_same_shapes(input, target, "cross_entropy_loss");
    if (input.rank() == 0)
        throw std::invalid_argument("cross_entropy_loss requires a non-scalar tensor");

    dim = tensor::Tensor<T>::normalize_dimension(dim, input.rank());

    const auto&   in_shape   = input.shape();
    const int64_t in_rank    = input.rank();
    const int64_t n          = input.numel();
    const auto    in_strides = tensor::Tensor<T>::compute_contiguous_strides(in_shape);

    // Build per-element slice index (identifies which "sample" each element belongs to)
    std::vector<int64_t> slice_shape;
    slice_shape.reserve(static_cast<size_t>(in_rank - 1));
    for (int64_t d = 0; d < in_rank; ++d)
        if (d != dim) slice_shape.push_back(in_shape[static_cast<size_t>(d)]);

    const int64_t n_batch      = tensor::Tensor<T>::compute_numel(slice_shape);
    const auto    slice_strides = tensor::Tensor<T>::compute_contiguous_strides(slice_shape);

    std::vector<int64_t> slice_of(static_cast<size_t>(n));
    for (int64_t flat = 0; flat < n; ++flat) {
        int64_t remaining  = flat;
        int64_t slice_flat = 0;
        int64_t slice_d    = 0;
        for (int64_t d = 0; d < in_rank; ++d) {
            const int64_t idx = remaining / in_strides[static_cast<size_t>(d)];
            remaining        %= in_strides[static_cast<size_t>(d)];
            if (d != dim) {
                slice_flat += idx * slice_strides[static_cast<size_t>(slice_d)];
                ++slice_d;
            }
        }
        slice_of[static_cast<size_t>(flat)] = slice_flat;
    }

    // pass 1: per-sample max for numerical stability
    std::vector<T> slice_max(static_cast<size_t>(n_batch), std::numeric_limits<T>::lowest());
    for (int64_t flat = 0; flat < n; ++flat) {
        T& m = slice_max[static_cast<size_t>(slice_of[static_cast<size_t>(flat)])];
        m    = std::max(m, input.data()[static_cast<size_t>(flat)]);
    }

    // pass 2: exp(x - max) and accumulate per-sample sum
    std::vector<T> softmax_storage(static_cast<size_t>(n));
    std::vector<T> slice_sum(static_cast<size_t>(n_batch), T{0});
    for (int64_t flat = 0; flat < n; ++flat) {
        const size_t s = static_cast<size_t>(slice_of[static_cast<size_t>(flat)]);
        softmax_storage[static_cast<size_t>(flat)] =
            static_cast<T>(std::exp(input.data()[static_cast<size_t>(flat)] - slice_max[s]));
        slice_sum[s] += softmax_storage[static_cast<size_t>(flat)];
    }

    // pass 3: normalise to get softmax
    for (int64_t flat = 0; flat < n; ++flat)
        softmax_storage[static_cast<size_t>(flat)] /=
            slice_sum[static_cast<size_t>(slice_of[static_cast<size_t>(flat)])];

    // pass 4: log_sum_exp per sample and accumulate NLL
    //   log_softmax(x_i) = x_i - slice_max - log(slice_sum)
    std::vector<T> log_sum_exp(static_cast<size_t>(n_batch));
    for (size_t s = 0; s < static_cast<size_t>(n_batch); ++s)
        log_sum_exp[s] = slice_max[s] + static_cast<T>(std::log(slice_sum[s]));

    T loss_sum = T{0};
    for (int64_t flat = 0; flat < n; ++flat) {
        const size_t s          = static_cast<size_t>(slice_of[static_cast<size_t>(flat)]);
        const T      log_soft   = input.data()[static_cast<size_t>(flat)] - log_sum_exp[s];
        loss_sum -= target.data()[static_cast<size_t>(flat)] * log_soft;
    }
    const T loss = loss_sum / static_cast<T>(n_batch);

    const bool requires_grad = autograd::is_grad_enabled()
                             && (input.requires_grad() || target.requires_grad());
    std::shared_ptr<autograd::Node<T>> grad_fn = nullptr;
    if (requires_grad) {
        const tensor::Tensor<T> softmax_saved = tensor::Tensor<T>::from_operation_result(
            in_shape, std::vector<T>(softmax_storage), false, nullptr);
        grad_fn = std::make_shared<autograd::CrossEntropyLossBackward<T>>(
            input, target, softmax_saved, n_batch);
    }

    return tensor::Tensor<T>::from_operation_result(
        {}, std::vector<T>{loss}, requires_grad, std::move(grad_fn));
}

// ----- bce_loss -----
// Binary cross-entropy (input must be probabilities in (0, 1)):
//   L = -mean(target * log(input) + (1 - target) * log(1 - input))
// Input is clamped to [eps, 1-eps] internally to avoid log(0).
// Returns a scalar tensor of shape {}.
template <typename T>
tensor::Tensor<T> bce_loss(const tensor::Tensor<T>& input, const tensor::Tensor<T>& target)
{
    check_same_shapes(input, target, "bce_loss");

    constexpr T eps = std::numeric_limits<T>::epsilon();
    T sum = T{0};
    for (int64_t i = 0; i < input.numel(); ++i) {
        const T p  = std::max(std::min(input.data()[static_cast<size_t>(i)],
                                       static_cast<T>(1) - eps), eps);
        const T t  = target.data()[static_cast<size_t>(i)];
        sum -= t * static_cast<T>(std::log(p))
             + (static_cast<T>(1) - t) * static_cast<T>(std::log(static_cast<T>(1) - p));
    }
    const T loss = sum / static_cast<T>(input.numel());

    const bool requires_grad = autograd::is_grad_enabled()
                             && (input.requires_grad() || target.requires_grad());
    std::shared_ptr<autograd::Node<T>> grad_fn = nullptr;
    if (requires_grad)
        grad_fn = std::make_shared<autograd::BCELossBackward<T>>(input, target);

    return tensor::Tensor<T>::from_operation_result(
        {}, std::vector<T>{loss}, requires_grad, std::move(grad_fn));
}

// ----- bce_with_logits_loss -----
// Numerically stable fused sigmoid + BCE (input is raw logits):
//   L = mean(log(1 + exp(input)) - input * target)
//     = mean(max(x,0) - x*t + log(1 + exp(-|x|)))
// Returns a scalar tensor of shape {}.
template <typename T>
tensor::Tensor<T> bce_with_logits_loss(const tensor::Tensor<T>& input,
                                        const tensor::Tensor<T>& target)
{
    check_same_shapes(input, target, "bce_with_logits_loss");

    T sum = T{0};
    for (int64_t i = 0; i < input.numel(); ++i) {
        const T x = input.data()[static_cast<size_t>(i)];
        const T t = target.data()[static_cast<size_t>(i)];
        // max(x, 0) - x*t + log(1 + exp(-|x|))
        const T pos = std::max(x, T{0});
        sum += pos - x * t + static_cast<T>(std::log(static_cast<T>(1)
                                             + static_cast<T>(std::exp(-std::abs(x)))));
    }
    const T loss = sum / static_cast<T>(input.numel());

    const bool requires_grad = autograd::is_grad_enabled()
                             && (input.requires_grad() || target.requires_grad());
    std::shared_ptr<autograd::Node<T>> grad_fn = nullptr;
    if (requires_grad)
        grad_fn = std::make_shared<autograd::BCEWithLogitsLossBackward<T>>(input, target);

    return tensor::Tensor<T>::from_operation_result(
        {}, std::vector<T>{loss}, requires_grad, std::move(grad_fn));
}

// ----- kl_div_loss -----
// Kullback-Leibler divergence (input must be log-probabilities, target probabilities):
//   L = mean(target * (log(target) - input))
// target is clamped to [eps, inf) to avoid log(0).
// Returns a scalar tensor of shape {}.
template <typename T>
tensor::Tensor<T> kl_div_loss(const tensor::Tensor<T>& input, const tensor::Tensor<T>& target)
{
    check_same_shapes(input, target, "kl_div_loss");

    constexpr T eps = std::numeric_limits<T>::epsilon();
    T sum = T{0};
    for (int64_t i = 0; i < input.numel(); ++i) {
        const T t     = target.data()[static_cast<size_t>(i)];
        const T log_t = static_cast<T>(std::log(std::max(t, eps)));
        sum += t * (log_t - input.data()[static_cast<size_t>(i)]);
    }
    const T loss = sum / static_cast<T>(input.numel());

    const bool requires_grad = autograd::is_grad_enabled()
                             && (input.requires_grad() || target.requires_grad());
    std::shared_ptr<autograd::Node<T>> grad_fn = nullptr;
    if (requires_grad)
        grad_fn = std::make_shared<autograd::KLDivLossBackward<T>>(input, target);

    return tensor::Tensor<T>::from_operation_result(
        {}, std::vector<T>{loss}, requires_grad, std::move(grad_fn));
}

} // namespace ops
