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

// y = (x - mean(x)) / sqrt(var(x) + eps) * weight + bias

template <typename T>
class LayerNorm : public Layer<T> {
public:
    tensor::Tensor<T> weight;
    tensor::Tensor<T> bias;
    double eps_;
    
    LayerNorm(int64_t num_features, double eps = 1e-5)
    : weight(tensor::Tensor<T>::random_gaussian(
          {num_features},
          T{0},
          static_cast<T>(1.0),
          true)),
      bias(tensor::Tensor<T>::zeros({num_features}, true)),
      eps_(eps)
    {}

    tensor::Tensor<T> forward(const tensor::Tensor<T>& input) const override {
        // Perform layer normalization over the last dimension
        const int64_t ndim = input.rank();
        if (ndim < 1)
            throw std::invalid_argument("LayerNorm: input rank must be at least 1");

        const int64_t feature_dim = input.shape().back();
        if (feature_dim != weight.size() || feature_dim != bias.size())
            throw std::invalid_argument("LayerNorm: Input feature dimension does not match weight/bias size");

        // Axes to reduce: all except the last axis
        std::vector<int64_t> axes(ndim - 1);
        for (int64_t i = 0; i < ndim - 1; ++i) axes[i] = i;

        // Compute mean and variance along the last axis
        auto mean = input.mean()
        auto variance = ops::mean(ops::pow(ops::sub(input, mean), 2), axes, /*keepdim=*/true);

        // Normalize
        auto normalized = ops::div(
            ops::sub(input, mean),
            ops::sqrt(ops::add(variance, T(eps_)))
        );

        // Scale and shift (weight and bias assumed shape {features})
        auto weight_bc = weight.broadcast_to(normalized.shape()); // stretch weight along batch axes
        auto bias_bc = bias.broadcast_to(normalized.shape());

        return ops::add(ops::mul(normalized, weight_bc), bias_bc);
    }

    std::vector<tensor::Tensor<T>*> parameters() {
        return {&weight, &bias};
    }
};

template <typename T>
class RMSNorm : public Layer<T> {
public:
    tensor::Tensor<T> weight;
    double eps_;
    
    RMSNorm(int64_t num_features, double eps = 1e-5)
    : weight(tensor::Tensor<T>::random_gaussian(
          {num_features},
          T{0},
          static_cast<T>(1.0),
          true)),
      eps_(eps)
    {}

    tensor::Tensor<T> forward(const tensor::Tensor<T>& input) const override {
        return input;
    }
};

} // namespace nn