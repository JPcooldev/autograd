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
#include "elementwise_ops.h"

namespace ops {

/**
 * Compute L1 loss. Packs both operands, then returns a scalar
 * `mean(|input - target|)` if `reduction` is true, otherwise
 * `sum(|input - target|)`. Attaches `L1LossBackward` when grad is enabled.
 *
 * @param input Predictions.
 * @param target Targets, same shape as `input`.
 * @param reduction If true, divide the sum by `numel`; otherwise return the sum.
 * @return A scalar tensor of shape `{}`.
 *
 * @throws std::invalid_argument if `input` and `target` have different shapes.
 */
template <typename T>
tensor::Tensor<T> l1_loss(
    const tensor::Tensor<T>& input,
    const tensor::Tensor<T>& target,
    bool reduction = true) {
    check_same_shapes(input, target, "l1_loss");
    const auto in_c  = contiguous(input);
    const auto tgt_c = contiguous(target);

    T sum = T{0};
    for (int64_t i = 0; i < in_c.numel(); ++i)
        sum += static_cast<T>(std::abs(
            in_c.data()[static_cast<size_t>(i)] - tgt_c.data()[static_cast<size_t>(i)]));
    const T loss = reduction ? sum / static_cast<T>(in_c.numel()) : sum;

    const bool requires_grad = autograd::is_grad_enabled() && (input.requires_grad() || target.requires_grad());
    std::shared_ptr<autograd::Node<T>> grad_fn = nullptr;
    if (requires_grad)
        grad_fn = std::make_shared<autograd::L1LossBackward<T>>(in_c, tgt_c, reduction);
    return tensor::Tensor<T>::from_operation_result(
        {}, std::vector<T>{loss}, requires_grad, std::move(grad_fn));
}

/**
 * Compute squared L2 loss. Packs both operands, then returns a scalar
 * `mean((input - target)^2)` if `reduction` is true, otherwise
 * `sum((input - target)^2)`. Attaches `L2LossBackward` when grad is enabled.
 *
 * @param input Predictions.
 * @param target Targets, same shape as `input`.
 * @param reduction If true, divide the sum by `numel`; otherwise return the sum.
 * @return A scalar tensor of shape `{}`.
 *
 * @throws std::invalid_argument if `input` and `target` have different shapes.
 */
template <typename T>
tensor::Tensor<T> l2_loss(
    const tensor::Tensor<T>& input,
    const tensor::Tensor<T>& target,
    bool reduction = true) {
    check_same_shapes(input, target, "l2_loss");
    const auto in_c  = contiguous(input);
    const auto tgt_c = contiguous(target);

    T sum = T{0};
    for (int64_t i = 0; i < in_c.numel(); ++i) {
        const T diff = in_c.data()[static_cast<size_t>(i)] - tgt_c.data()[static_cast<size_t>(i)];
        sum += diff * diff;
    }
    const T loss = reduction ? sum / static_cast<T>(in_c.numel()) : sum;

    const bool requires_grad = autograd::is_grad_enabled() && (input.requires_grad() || target.requires_grad());
    std::shared_ptr<autograd::Node<T>> grad_fn = nullptr;
    if (requires_grad)
        grad_fn = std::make_shared<autograd::L2LossBackward<T>>(in_c, tgt_c, reduction);
    return tensor::Tensor<T>::from_operation_result(
        {}, std::vector<T>{loss}, requires_grad, std::move(grad_fn));
}

/**
 * Compute mean squared error. Delegates to `l2_loss` (mean reduction by default).
 *
 * @param input Predictions.
 * @param target Targets, same shape as `input`.
 * @param reduction If true, divide the sum by `numel`; otherwise return the sum.
 * @return A scalar tensor of shape `{}`.
 *
 * @throws std::invalid_argument if `input` and `target` have different shapes.
 */
template <typename T>
tensor::Tensor<T> mse_loss(
    const tensor::Tensor<T>& input,
    const tensor::Tensor<T>& target,
    bool reduction = true) {
    return l2_loss(input, target, reduction);
}

/**
 * Compute negative log-likelihood loss. Input must be log-probabilities;
 * `L = -(1/N_batch) * sum target * input` where `N_batch = numel / shape[dim]`.
 * Attaches `NLLLossBackward` when grad is enabled.
 *
 * @param input Log-probabilities, same shape as `target`.
 * @param target Soft labels (or one-hot), same shape as `input`.
 * @param dim Class axis (negative indices allowed; default -1).
 * @return A scalar tensor of shape `{}`.
 *
 * @throws std::invalid_argument if `input` and `target` have different shapes.
 * @throws std::invalid_argument if `input` is empty.
 * @throws std::out_of_range if `dim` is out of range.
 */
template <typename T>
tensor::Tensor<T> nll_loss(
    const tensor::Tensor<T>& input,
    const tensor::Tensor<T>& target,
    int64_t dim = -1) {
    check_same_shapes(input, target, "nll_loss");
    if (input.numel() == 0)
        throw std::invalid_argument("nll_loss requires a non-empty tensor");

    dim = tensor::Tensor<T>::normalize_dimension(dim, input.rank());
    const auto in_c  = contiguous(input);
    const auto tgt_c = contiguous(target);
    const int64_t n_batch = in_c.numel() / in_c.shape()[static_cast<size_t>(dim)];

    T sum = T{0};
    for (int64_t i = 0; i < in_c.numel(); ++i)
        sum -= tgt_c.data()[static_cast<size_t>(i)] * in_c.data()[static_cast<size_t>(i)];
    const T loss = sum / static_cast<T>(n_batch);

    const bool requires_grad = autograd::is_grad_enabled() && (input.requires_grad() || target.requires_grad());
    std::shared_ptr<autograd::Node<T>> grad_fn = nullptr;
    if (requires_grad)
        grad_fn = std::make_shared<autograd::NLLLossBackward<T>>(in_c, tgt_c, n_batch);
    return tensor::Tensor<T>::from_operation_result(
        {}, std::vector<T>{loss}, requires_grad, std::move(grad_fn));
}

/**
 * Compute fused log-softmax + NLL (numerically stable).
 * `L = -(1/N_batch) * sum target * log_softmax(input, dim)` with a per-sample
 * max-subtraction. Attaches `CrossEntropyLossBackward` when grad is enabled.
 *
 * @param input Logits, same shape as `target`.
 * @param target Soft labels (probabilities), same shape as `input`.
 * @param dim Class axis (negative indices allowed; default -1).
 * @return A scalar tensor of shape `{}`.
 *
 * @throws std::invalid_argument if `input` and `target` have different shapes.
 * @throws std::invalid_argument if `input` is a scalar.
 * @throws std::out_of_range if `dim` is out of range.
 */
template <typename T>
tensor::Tensor<T> cross_entropy_loss(
    const tensor::Tensor<T>& input,
    const tensor::Tensor<T>& target,
    int64_t dim = -1) {
    check_same_shapes(input, target, "cross_entropy_loss");
    if (input.rank() == 0)
        throw std::invalid_argument("cross_entropy_loss requires a non-scalar tensor");

    dim = tensor::Tensor<T>::normalize_dimension(dim, input.rank());
    const auto in_c  = contiguous(input);
    const auto tgt_c = contiguous(target);

    const auto&   in_shape   = in_c.shape();
    const int64_t in_rank    = in_c.rank();
    const int64_t n          = in_c.numel();
    const auto    in_strides = tensor::Tensor<T>::compute_contiguous_strides(in_shape);

    std::vector<int64_t> slice_shape;
    slice_shape.reserve(static_cast<size_t>(in_rank - 1));
    for (int64_t d = 0; d < in_rank; ++d)
        if (d != dim)
            slice_shape.push_back(in_shape[static_cast<size_t>(d)]);

    const int64_t n_batch       = tensor::Tensor<T>::compute_numel(slice_shape);
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

    std::vector<T> slice_max(static_cast<size_t>(n_batch), std::numeric_limits<T>::lowest());
    for (int64_t flat = 0; flat < n; ++flat) {
        T& m = slice_max[static_cast<size_t>(slice_of[static_cast<size_t>(flat)])];
        m    = std::max(m, in_c.data()[static_cast<size_t>(flat)]);
    }

    std::vector<T> softmax_storage(static_cast<size_t>(n));
    std::vector<T> slice_sum(static_cast<size_t>(n_batch), T{0});
    for (int64_t flat = 0; flat < n; ++flat) {
        const size_t s = static_cast<size_t>(slice_of[static_cast<size_t>(flat)]);
        softmax_storage[static_cast<size_t>(flat)] =
            static_cast<T>(std::exp(in_c.data()[static_cast<size_t>(flat)] - slice_max[s]));
        slice_sum[s] += softmax_storage[static_cast<size_t>(flat)];
    }

    for (int64_t flat = 0; flat < n; ++flat)
        softmax_storage[static_cast<size_t>(flat)] /=
            slice_sum[static_cast<size_t>(slice_of[static_cast<size_t>(flat)])];

    std::vector<T> log_sum_exp(static_cast<size_t>(n_batch));
    for (size_t s = 0; s < static_cast<size_t>(n_batch); ++s)
        log_sum_exp[s] = slice_max[s] + static_cast<T>(std::log(slice_sum[s]));

    T loss_sum = T{0};
    for (int64_t flat = 0; flat < n; ++flat) {
        const size_t s        = static_cast<size_t>(slice_of[static_cast<size_t>(flat)]);
        const T      log_soft = in_c.data()[static_cast<size_t>(flat)] - log_sum_exp[s];
        loss_sum -= tgt_c.data()[static_cast<size_t>(flat)] * log_soft;
    }
    const T loss = loss_sum / static_cast<T>(n_batch);

    const bool requires_grad = autograd::is_grad_enabled()
                             && (input.requires_grad() || target.requires_grad());
    std::shared_ptr<autograd::Node<T>> grad_fn = nullptr;
    if (requires_grad) {
        const tensor::Tensor<T> softmax_saved = tensor::Tensor<T>::from_operation_result(
            in_shape, std::vector<T>(softmax_storage), false, nullptr);
        grad_fn = std::make_shared<autograd::CrossEntropyLossBackward<T>>(
            in_c, tgt_c, softmax_saved, n_batch);
    }

    return tensor::Tensor<T>::from_operation_result(
        {}, std::vector<T>{loss}, requires_grad, std::move(grad_fn));
}

/**
 * Compute binary cross-entropy. Input is treated as probabilities and clamped
 * to `[eps, 1-eps]` before `L = -mean(t * log(p) + (1-t) * log(1-p))`.
 * Attaches `BCELossBackward` when grad is enabled.
 *
 * @param input Probabilities, same shape as `target`.
 * @param target Targets in `[0, 1]`, same shape as `input`.
 * @return A scalar tensor of shape `{}`.
 *
 * @throws std::invalid_argument if `input` and `target` have different shapes.
 */
template <typename T>
tensor::Tensor<T> bce_loss(
    const tensor::Tensor<T>& input,
    const tensor::Tensor<T>& target) {
    check_same_shapes(input, target, "bce_loss");
    const auto in_c  = contiguous(input);
    const auto tgt_c = contiguous(target);

    constexpr T eps = std::numeric_limits<T>::epsilon();
    T sum = T{0};
    for (int64_t i = 0; i < in_c.numel(); ++i) {
        const T p = std::max(std::min(in_c.data()[static_cast<size_t>(i)], static_cast<T>(1) - eps), eps);
        const T t = tgt_c.data()[static_cast<size_t>(i)];
        sum -= t * static_cast<T>(std::log(p))
             + (static_cast<T>(1) - t) * static_cast<T>(std::log(static_cast<T>(1) - p));
    }
    const T loss = sum / static_cast<T>(in_c.numel());

    const bool requires_grad = autograd::is_grad_enabled() && (input.requires_grad() || target.requires_grad());
    std::shared_ptr<autograd::Node<T>> grad_fn = nullptr;
    if (requires_grad)
        grad_fn = std::make_shared<autograd::BCELossBackward<T>>(in_c, tgt_c);
    return tensor::Tensor<T>::from_operation_result(
        {}, std::vector<T>{loss}, requires_grad, std::move(grad_fn));
}

/**
 * Compute fused sigmoid + BCE from logits. Uses the stable form
 * `mean(max(x,0) - x*t + log(1 + exp(-|x|)))`. Attaches
 * `BCEWithLogitsLossBackward` when grad is enabled.
 *
 * @param input Logits, same shape as `target`.
 * @param target Targets in `[0, 1]`, same shape as `input`.
 * @return A scalar tensor of shape `{}`.
 *
 * @throws std::invalid_argument if `input` and `target` have different shapes.
 */
template <typename T>
tensor::Tensor<T> bce_with_logits_loss(
    const tensor::Tensor<T>& input,
    const tensor::Tensor<T>& target) {
    check_same_shapes(input, target, "bce_with_logits_loss");
    const auto in_c  = contiguous(input);
    const auto tgt_c = contiguous(target);

    T sum = T{0};
    for (int64_t i = 0; i < in_c.numel(); ++i) {
        const T x = in_c.data()[static_cast<size_t>(i)];
        const T t = tgt_c.data()[static_cast<size_t>(i)];
        const T pos = std::max(x, T{0});
        sum += pos - x * t + static_cast<T>(std::log(static_cast<T>(1)
                                             + static_cast<T>(std::exp(-std::abs(x)))));
    }
    const T loss = sum / static_cast<T>(in_c.numel());

    const bool requires_grad = autograd::is_grad_enabled() && (input.requires_grad() || target.requires_grad());
    std::shared_ptr<autograd::Node<T>> grad_fn = nullptr;
    if (requires_grad)
        grad_fn = std::make_shared<autograd::BCEWithLogitsLossBackward<T>>(in_c, tgt_c);

    return tensor::Tensor<T>::from_operation_result(
        {}, std::vector<T>{loss}, requires_grad, std::move(grad_fn));
}

/**
 * Compute Kullback–Leibler divergence. Input is log-probabilities, target is
 * probabilities; `L = mean(target * (log(target) - input))` with `target`
 * clamped to `[eps, inf)`. Attaches `KLDivLossBackward` when grad is enabled.
 *
 * @param input Log-probabilities, same shape as `target`.
 * @param target Probabilities, same shape as `input`.
 * @return A scalar tensor of shape `{}`.
 *
 * @throws std::invalid_argument if `input` and `target` have different shapes.
 */
template <typename T>
tensor::Tensor<T> kl_div_loss(
    const tensor::Tensor<T>& input,
    const tensor::Tensor<T>& target) {
    check_same_shapes(input, target, "kl_div_loss");
    const auto in_c  = contiguous(input);
    const auto tgt_c = contiguous(target);

    constexpr T eps = std::numeric_limits<T>::epsilon();
    T sum = T{0};
    for (int64_t i = 0; i < in_c.numel(); ++i) {
        const T t     = tgt_c.data()[static_cast<size_t>(i)];
        const T log_t = static_cast<T>(std::log(std::max(t, eps)));
        sum += t * (log_t - in_c.data()[static_cast<size_t>(i)]);
    }
    const T loss = sum / static_cast<T>(in_c.numel());

    const bool requires_grad = autograd::is_grad_enabled() && (input.requires_grad() || target.requires_grad());
    std::shared_ptr<autograd::Node<T>> grad_fn = nullptr;
    if (requires_grad)
        grad_fn = std::make_shared<autograd::KLDivLossBackward<T>>(in_c, tgt_c);
    return tensor::Tensor<T>::from_operation_result(
        {}, std::vector<T>{loss}, requires_grad, std::move(grad_fn));
}

} // namespace ops
