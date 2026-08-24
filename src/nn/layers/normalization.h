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

template <typename T>
tensor::Tensor<T> broadcast_feature(const tensor::Tensor<T>& feat, const tensor::Tensor<T>& like)
{
    auto t = feat;
    while (t.rank() < like.rank())
        t = t.unsqueeze(0);
    return t.broadcast_to(like.shape()).contiguous();
}

template <typename T>
tensor::Tensor<T> last_dim_mean(const tensor::Tensor<T>& x)
{
    return x.mean(-1).unsqueeze(-1).broadcast_to(x.shape()).contiguous();
}

} // namespace norm_detail

// y = (x - μ) / sqrt(σ² + ε) * γ + β   over the last axis
template <typename T>
class LayerNorm : public Layer<T> {
public:
    tensor::Tensor<T> weight;
    tensor::Tensor<T> bias;
    double eps;

    LayerNorm(int64_t num_features, double eps_ = 1e-5)
        : weight(tensor::Tensor<T>::ones({num_features}, true)),
          bias(tensor::Tensor<T>::zeros({num_features}, true)),
          eps(eps_),
          num_features_(num_features)
    {}

    tensor::Tensor<T> forward(const tensor::Tensor<T>& input) const
    {
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

    std::vector<tensor::Tensor<T>*> parameters() override { return {&weight, &bias}; }

private:
    int64_t num_features_;
};

// y = x / sqrt(mean(x²) + ε) * γ
template <typename T>
class RMSNorm : public Layer<T> {
public:
    tensor::Tensor<T> weight;
    double eps;

    RMSNorm(int64_t num_features, double eps_ = 1e-5)
        : weight(tensor::Tensor<T>::ones({num_features}, true)),
          eps(eps_),
          num_features_(num_features)
    {}

    tensor::Tensor<T> forward(const tensor::Tensor<T>& input) const
    {
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

    std::vector<tensor::Tensor<T>*> parameters() override { return {&weight}; }

private:
    int64_t num_features_;
};

template <typename T>
class BatchNorm : public Layer<T> {
public:
    tensor::Tensor<T> weight;
    tensor::Tensor<T> bias;
    tensor::Tensor<T> running_mean;
    tensor::Tensor<T> running_var;
    double eps;
    double momentum;

    BatchNorm(int64_t num_features, double eps_ = 1e-5, double momentum_ = 0.1)
        : weight(tensor::Tensor<T>::ones({num_features}, true)),
          bias(tensor::Tensor<T>::zeros({num_features}, true)),
          running_mean(tensor::Tensor<T>::zeros({num_features}, false)),
          running_var(tensor::Tensor<T>::ones({num_features}, false)),
          eps(eps_),
          momentum(momentum_),
          num_features_(num_features)
    {}

    tensor::Tensor<T> forward(const tensor::Tensor<T>& input)
    {
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

    std::vector<tensor::Tensor<T>*> parameters() override { return {&weight, &bias}; }

private:
    int64_t num_features_;
};

template <typename T>
class BatchNorm1d : public BatchNorm<T> {
public:
    using BatchNorm<T>::BatchNorm;
};

template <typename T>
class BatchNorm2d : public BatchNorm<T> {
public:
    using BatchNorm<T>::BatchNorm;
};

template <typename T>
class BatchNorm3d : public BatchNorm<T> {
public:
    using BatchNorm<T>::BatchNorm;
};

} // namespace nn
