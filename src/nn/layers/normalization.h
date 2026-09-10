#pragma once

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "../../tensor/tensor.h"
#include "layer.h"
#include "../../ops/elementwise_ops.h"
#include "../../ops/reduction_ops.h"
#include "../../ops/shape_ops.h"

namespace nn {

namespace norm_detail {

/**
 * Broadcast a rank-1 feature vector onto `like`.
 * Unsqueeze leading dims until ranks match, then `broadcast_to(like.shape())` and `contiguous()`.
 *
 * @param feat Feature tensor, typically shape `{C}`.
 * @param like Tensor whose shape is the broadcast target.
 * @return `feat` broadcast and made contiguous at `like`'s shape.
 */
template <typename T>
tensor::Tensor<T> broadcast_feature(const tensor::Tensor<T>& feat, const tensor::Tensor<T>& like) {
    auto t = feat;
    while (t.rank() < like.rank())
        t = t.unsqueeze(0);
    return t.broadcast_to(like.shape()).contiguous();
}

/**
 * Repeat the last-axis mean of `x` back to `x`'s shape.
 * Computes `x.mean(-1)`, unsqueezes the last dim, broadcasts, and copies to contiguous storage.
 *
 * @param x Input tensor with rank ≥ 1.
 * @return Tensor with the same shape as `x`, each position holding the mean of its last-axis slice.
 */
template <typename T>
tensor::Tensor<T> last_dim_mean(const tensor::Tensor<T>& x) {
    return x.mean(-1).unsqueeze(-1).broadcast_to(x.shape()).contiguous();
}

} // namespace norm_detail

// y = (x - μ) / sqrt(σ² + ε) * γ + β   over the last axis
template <typename T>
class LayerNorm : public Layer<T> {
private:
    int64_t num_features_;

public:
    tensor::Tensor<T> weight;
    tensor::Tensor<T> bias;
    double eps;

    /**
     * Construct last-axis LayerNorm.
     * Initializes `weight` (γ) to ones and `bias` (β) to zeros, both shape `{num_features}` with `requires_grad`.
     *
     * @param num_features Size of the last dimension of `forward` inputs.
     * @param eps_ Denominator stabilizer ε. Default 1e-5.
     */
    LayerNorm(int64_t num_features, double eps_ = 1e-5)
        : num_features_(num_features),
          weight(tensor::Tensor<T>::ones({num_features}, true)),
          bias(tensor::Tensor<T>::zeros({num_features}, true)),
          eps(eps_) {}

    /**
     * Apply affine LayerNorm over the last axis: `(x-μ)/√(σ²+ε) * γ + β`.
     * Uses batch-independent mean/var along the last dimension; `weight` and `bias` are broadcast to `x`.
     *
     * @param input Tensor of shape `(..., num_features)`.
     * @return Normalized tensor of the same shape.
     *
     * @throws std::invalid_argument if `input.rank() < 1`.
     * @throws std::invalid_argument if the last dimension is not `num_features`.
     */
    tensor::Tensor<T> forward(const tensor::Tensor<T>& input) const {
        if (input.rank() < 1)
            throw std::invalid_argument("LayerNorm: input rank must be >= 1");
        if (input.shape().back() != num_features_)
            throw std::invalid_argument("LayerNorm: last dim does not match num_features");

        const auto x = input.contiguous();
        const auto mu = norm_detail::last_dim_mean(x);
        const auto diff = ops::subtract(x, mu);
        const auto var = norm_detail::last_dim_mean(ops::multiply(diff, diff));
        const auto eps_t = tensor::Tensor<T>::full(var.shape(), static_cast<T>(eps), false);
        const auto y = ops::divide(diff, ops::sqrt(ops::add(var, eps_t)));
        return ops::add(
            ops::multiply(y, norm_detail::broadcast_feature(weight, y)),
            norm_detail::broadcast_feature(bias, y));
    }

    /**
     * Return named pointers to γ and β.
     *
     * @return `{{"weight", &weight}, {"bias", &bias}}`.
     */
    NamedTensorList<T> named_parameters() override {
        return {{"weight", &weight}, {"bias", &bias}};
    }
};

// y = x / sqrt(mean(x²) + ε) * γ
template <typename T>
class RMSNorm : public Layer<T> {
private:
    int64_t num_features_;

public:
    tensor::Tensor<T> weight;
    double eps;

    /**
     * Construct last-axis RMSNorm.
     * Initializes `weight` (γ) to ones of shape `{num_features}` with `requires_grad`. No bias.
     *
     * @param num_features Size of the last dimension of `forward` inputs.
     * @param eps_ Denominator stabilizer ε. Default 1e-5.
     */
    RMSNorm(int64_t num_features, double eps_ = 1e-5)
        : num_features_(num_features),
          weight(tensor::Tensor<T>::ones({num_features}, true)),
          eps(eps_) {}

    /**
     * Apply RMSNorm over the last axis: `x / √(mean(x²)+ε) * γ`.
     * `weight` is broadcast to the normalized tensor.
     *
     * @param input Tensor of shape `(..., num_features)`.
     * @return Normalized tensor of the same shape.
     *
     * @throws std::invalid_argument if `input.rank() < 1`.
     * @throws std::invalid_argument if the last dimension is not `num_features`.
     */
    tensor::Tensor<T> forward(const tensor::Tensor<T>& input) const {
        if (input.rank() < 1)
            throw std::invalid_argument("RMSNorm: input rank must be >= 1");
        if (input.shape().back() != num_features_)
            throw std::invalid_argument("RMSNorm: last dim does not match num_features");

        const auto x = input.contiguous();
        const auto ms = norm_detail::last_dim_mean(ops::multiply(x, x));
        const auto eps_t = tensor::Tensor<T>::full(ms.shape(), static_cast<T>(eps), false);
        const auto y = ops::divide(x, ops::sqrt(ops::add(ms, eps_t)));
        return ops::multiply(y, norm_detail::broadcast_feature(weight, y));
    }

    /**
     * Return a named pointer to γ.
     *
     * @return `{{"weight", &weight}}`.
     */
    NamedTensorList<T> named_parameters() override { return {{"weight", &weight}}; }
};

template <typename T>
class BatchNorm : public Layer<T> {
private:
    int64_t num_features_;

public:
    tensor::Tensor<T> weight;
    tensor::Tensor<T> bias;
    tensor::Tensor<T> running_mean;
    tensor::Tensor<T> running_var;
    double eps;
    double momentum;

    /**
     * Construct channel-axis batch normalization (dim 1).
     * Initializes γ to ones, β to zeros (`requires_grad`); running mean to zeros and running var
     * to ones (`requires_grad=false`).
     *
     * @param num_features Channel count C (input dim 1).
     * @param eps_ Denominator stabilizer ε. Default 1e-5.
     * @param momentum_ Factor m for `running = (1-m)·running + m·batch` in training. Default 0.1.
     */
    BatchNorm(int64_t num_features, double eps_ = 1e-5, double momentum_ = 0.1)
        : num_features_(num_features),
          weight(tensor::Tensor<T>::ones({num_features}, true)),
          bias(tensor::Tensor<T>::zeros({num_features}, true)),
          running_mean(tensor::Tensor<T>::zeros({num_features}, false)),
          running_var(tensor::Tensor<T>::ones({num_features}, false)),
          eps(eps_),
          momentum(momentum_) {}

    /**
     * Normalize over all axes except channel dim 1, then apply affine γ, β.
     * Training: uses batch mean/var and updates running stats with `momentum`.
     * Eval: uses `running_mean` / `running_var` only. Non-const because running buffers are written in train.
     *
     * @param input Tensor of shape `(N, C, ...)` with `C == num_features`.
     * @return Normalized tensor of the same shape.
     *
     * @throws std::invalid_argument if `input.rank() < 2`.
     * @throws std::invalid_argument if dim 1 is not `num_features`.
     */
    tensor::Tensor<T> forward(const tensor::Tensor<T>& input) {
        if (input.rank() < 2)
            throw std::invalid_argument("BatchNorm: expected (N, C, ...)");
        if (input.shape()[1] != num_features_)
            throw std::invalid_argument("BatchNorm: channel dim does not match num_features");

        const auto x = input.contiguous();
        tensor::Tensor<T> mean = running_mean;
        tensor::Tensor<T> var = running_var;

        if (this->training()) {
            auto xf = x.rank() > 2 ? x.flatten(2, -1) : x.unsqueeze(-1);
            // xf: (N, C, spat)
            auto mu_cs = xf.mean(0);          // (C, spat)
            auto mu = mu_cs.mean(-1);         // (C,)
            auto mu_b = mu.unsqueeze(0).unsqueeze(-1).broadcast_to(xf.shape());
            auto diff = ops::subtract(xf, mu_b);
            auto var_cs = ops::mean(ops::multiply(diff, diff), 0); // (C, spat)
            auto v = var_cs.mean(-1); // (C,)
            mean = mu;
            var = v;

            auto& rm = running_mean.data();
            auto& rv = running_var.data();
            const auto& md = mu.data();
            const auto& vd = v.data();
            const T m = static_cast<T>(momentum);
            const T om = static_cast<T>(1) - m;
            for (int64_t i = 0; i < num_features_; ++i) {
                rm[static_cast<size_t>(i)] = om * rm[static_cast<size_t>(i)] + m * md[static_cast<size_t>(i)];
                rv[static_cast<size_t>(i)] = om * rv[static_cast<size_t>(i)] + m * vd[static_cast<size_t>(i)];
            }
        }

        // mean/var are rank-1 (C,). Broadcast to (1, C, 1, ..., 1) then to x.
        auto m = mean.unsqueeze(0);
        auto v = var.unsqueeze(0);
        while (m.rank() < x.rank()) {
            m = m.unsqueeze(-1);
            v = v.unsqueeze(-1);
        }
        m = m.broadcast_to(x.shape()).contiguous();
        v = v.broadcast_to(x.shape()).contiguous();
        const auto eps_t = tensor::Tensor<T>::full(v.shape(), static_cast<T>(eps), false);
        const auto y = ops::divide(ops::subtract(x, m), ops::sqrt(ops::add(v, eps_t)));
        auto w = weight.unsqueeze(0);
        auto b = bias.unsqueeze(0);
        while (w.rank() < x.rank()) {
            w = w.unsqueeze(-1);
            b = b.unsqueeze(-1);
        }
        return ops::add(
            ops::multiply(y, w.broadcast_to(x.shape()).contiguous()),
            b.broadcast_to(x.shape()).contiguous());
    }

    /**
     * Return named pointers to γ and β (not running buffers).
     *
     * @return `{{"weight", &weight}, {"bias", &bias}}`.
     */
    NamedTensorList<T> named_parameters() override {
        return {{"weight", &weight}, {"bias", &bias}};
    }

    /**
     * Return named running statistics used at eval time.
     *
     * @return `{{"running_mean", &running_mean}, {"running_var", &running_var}}`.
     */
    NamedTensorList<T> named_buffers() override {
        return {{"running_mean", &running_mean}, {"running_var", &running_var}};
    }
};

template <typename T>
class BatchNorm1d : public BatchNorm<T> {
public:
    /**
     * Inherit `BatchNorm` constructors (`num_features`, `eps=1e-5`, `momentum=0.1`).
     * Same forward as `BatchNorm` for `(N, C)` or `(N, C, L)`.
     */
    using BatchNorm<T>::BatchNorm;
};

template <typename T>
class BatchNorm2d : public BatchNorm<T> {
public:
    /**
     * Inherit `BatchNorm` constructors (`num_features`, `eps=1e-5`, `momentum=0.1`).
     * Same forward as `BatchNorm` for `(N, C, H, W)`.
     */
    using BatchNorm<T>::BatchNorm;
};

template <typename T>
class BatchNorm3d : public BatchNorm<T> {
public:
    /**
     * Inherit `BatchNorm` constructors (`num_features`, `eps=1e-5`, `momentum=0.1`).
     * Same forward as `BatchNorm` for `(N, C, D, H, W)`.
     */
    using BatchNorm<T>::BatchNorm;
};

} // namespace nn
