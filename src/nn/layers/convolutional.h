#pragma once

#include <cmath>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <vector>

#include "../../tensor/tensor.h"
#include "layer.h"
#include "../../ops/conv_ops.h"

namespace nn {

namespace conv_init {

template <typename T>
void kaiming_weight_and_bias(
    tensor::Tensor<T>& weight,
    std::optional<tensor::Tensor<T>>& bias,
    int64_t out_channels,
    bool use_bias)
{
    weight = tensor::Tensor<T>::kaiming_uniform(
        weight.shape(), std::sqrt(5.0), "fan_in", true);
    if (use_bias) {
        auto [fan_in, fan_out] = tensor::Tensor<T>::compute_fans(weight.shape());
        (void)fan_out;
        const T bound = static_cast<T>(1.0 / std::sqrt(static_cast<double>(fan_in)));
        bias = tensor::Tensor<T>::uniform({out_channels}, -bound, bound, true);
    }
}

} // namespace conv_init

// ---------------------------------------------------------------------------
// Conv1d / Conv2d / Conv3d
// ---------------------------------------------------------------------------
// y = conv(x, W) + b     W shape {out_channels, in_channels, *kernel}

template <typename T>
class Conv1d : public Layer<T> {
public:
    tensor::Tensor<T> weight;
    std::optional<tensor::Tensor<T>> bias;

    Conv1d(int64_t in_channels, int64_t out_channels, int64_t kernel_size,
           int64_t stride = 1, int64_t padding = 0, int64_t dilation = 1,
           bool use_bias = true)
        : weight(tensor::Tensor<T>::zeros({out_channels, in_channels, kernel_size}, true)),
          stride_(stride), padding_(padding), dilation_(dilation),
          in_channels_(in_channels), out_channels_(out_channels), kernel_size_(kernel_size)
    {
        conv_init::kaiming_weight_and_bias(weight, bias, out_channels, use_bias);
    }

    tensor::Tensor<T> forward(const tensor::Tensor<T>& input) const
    {
        return ops::conv1d(input, weight, bias ? &*bias : nullptr, stride_, padding_, dilation_);
    }

    std::vector<tensor::Tensor<T>*> parameters() override
    {
        if (bias) return {&weight, &(*bias)};
        return {&weight};
    }

private:
    int64_t stride_, padding_, dilation_;
    int64_t in_channels_, out_channels_, kernel_size_;
};

template <typename T>
class Conv2d : public Layer<T> {
public:
    tensor::Tensor<T> weight;
    std::optional<tensor::Tensor<T>> bias;

    Conv2d(int64_t in_channels, int64_t out_channels, int64_t kernel_size,
           int64_t stride = 1, int64_t padding = 0, int64_t dilation = 1,
           bool use_bias = true)
        : weight(tensor::Tensor<T>::zeros(
              {out_channels, in_channels, kernel_size, kernel_size}, true)),
          stride_(stride), padding_(padding), dilation_(dilation),
          in_channels_(in_channels), out_channels_(out_channels), kernel_size_(kernel_size)
    {
        conv_init::kaiming_weight_and_bias(weight, bias, out_channels, use_bias);
    }

    tensor::Tensor<T> forward(const tensor::Tensor<T>& input) const
    {
        return ops::conv2d(input, weight, bias ? &*bias : nullptr, stride_, padding_, dilation_);
    }

    std::vector<tensor::Tensor<T>*> parameters() override
    {
        if (bias) return {&weight, &(*bias)};
        return {&weight};
    }

private:
    int64_t stride_, padding_, dilation_;
    int64_t in_channels_, out_channels_, kernel_size_;
};

template <typename T>
class Conv3d : public Layer<T> {
public:
    tensor::Tensor<T> weight;
    std::optional<tensor::Tensor<T>> bias;

    Conv3d(int64_t in_channels, int64_t out_channels, int64_t kernel_size,
           int64_t stride = 1, int64_t padding = 0, int64_t dilation = 1,
           bool use_bias = true)
        : weight(tensor::Tensor<T>::zeros(
              {out_channels, in_channels, kernel_size, kernel_size, kernel_size}, true)),
          stride_(stride), padding_(padding), dilation_(dilation),
          in_channels_(in_channels), out_channels_(out_channels), kernel_size_(kernel_size)
    {
        conv_init::kaiming_weight_and_bias(weight, bias, out_channels, use_bias);
    }

    tensor::Tensor<T> forward(const tensor::Tensor<T>& input) const
    {
        return ops::conv3d(input, weight, bias ? &*bias : nullptr, stride_, padding_, dilation_);
    }

    std::vector<tensor::Tensor<T>*> parameters() override
    {
        if (bias) return {&weight, &(*bias)};
        return {&weight};
    }

private:
    int64_t stride_, padding_, dilation_;
    int64_t in_channels_, out_channels_, kernel_size_;
};

// ---------------------------------------------------------------------------
// ConvTranspose1d / 2d / 3d
// ---------------------------------------------------------------------------
// W shape {in_channels, out_channels, *kernel}

template <typename T>
class ConvTranspose1d : public Layer<T> {
public:
    tensor::Tensor<T> weight;
    std::optional<tensor::Tensor<T>> bias;

    ConvTranspose1d(int64_t in_channels, int64_t out_channels, int64_t kernel_size,
                    int64_t stride = 1, int64_t padding = 0, int64_t dilation = 1,
                    bool use_bias = true)
        : weight(tensor::Tensor<T>::zeros({in_channels, out_channels, kernel_size}, true)),
          stride_(stride), padding_(padding), dilation_(dilation),
          in_channels_(in_channels), out_channels_(out_channels), kernel_size_(kernel_size)
    {
        conv_init::kaiming_weight_and_bias(weight, bias, out_channels, use_bias);
    }

    tensor::Tensor<T> forward(const tensor::Tensor<T>& input) const
    {
        return ops::conv_transpose1d(
            input, weight, bias ? &*bias : nullptr, stride_, padding_, dilation_);
    }

    std::vector<tensor::Tensor<T>*> parameters() override
    {
        if (bias) return {&weight, &(*bias)};
        return {&weight};
    }

private:
    int64_t stride_, padding_, dilation_;
    int64_t in_channels_, out_channels_, kernel_size_;
};

template <typename T>
class ConvTranspose2d : public Layer<T> {
public:
    tensor::Tensor<T> weight;
    std::optional<tensor::Tensor<T>> bias;

    ConvTranspose2d(int64_t in_channels, int64_t out_channels, int64_t kernel_size,
                    int64_t stride = 1, int64_t padding = 0, int64_t dilation = 1,
                    bool use_bias = true)
        : weight(tensor::Tensor<T>::zeros(
              {in_channels, out_channels, kernel_size, kernel_size}, true)),
          stride_(stride), padding_(padding), dilation_(dilation),
          in_channels_(in_channels), out_channels_(out_channels), kernel_size_(kernel_size)
    {
        conv_init::kaiming_weight_and_bias(weight, bias, out_channels, use_bias);
    }

    tensor::Tensor<T> forward(const tensor::Tensor<T>& input) const
    {
        return ops::conv_transpose2d(
            input, weight, bias ? &*bias : nullptr, stride_, padding_, dilation_);
    }

    std::vector<tensor::Tensor<T>*> parameters() override
    {
        if (bias) return {&weight, &(*bias)};
        return {&weight};
    }

private:
    int64_t stride_, padding_, dilation_;
    int64_t in_channels_, out_channels_, kernel_size_;
};

template <typename T>
class ConvTranspose3d : public Layer<T> {
public:
    tensor::Tensor<T> weight;
    std::optional<tensor::Tensor<T>> bias;

    ConvTranspose3d(int64_t in_channels, int64_t out_channels, int64_t kernel_size,
                    int64_t stride = 1, int64_t padding = 0, int64_t dilation = 1,
                    bool use_bias = true)
        : weight(tensor::Tensor<T>::zeros(
              {in_channels, out_channels, kernel_size, kernel_size, kernel_size}, true)),
          stride_(stride), padding_(padding), dilation_(dilation),
          in_channels_(in_channels), out_channels_(out_channels), kernel_size_(kernel_size)
    {
        conv_init::kaiming_weight_and_bias(weight, bias, out_channels, use_bias);
    }

    tensor::Tensor<T> forward(const tensor::Tensor<T>& input) const
    {
        return ops::conv_transpose3d(
            input, weight, bias ? &*bias : nullptr, stride_, padding_, dilation_);
    }

    std::vector<tensor::Tensor<T>*> parameters() override
    {
        if (bias) return {&weight, &(*bias)};
        return {&weight};
    }

private:
    int64_t stride_, padding_, dilation_;
    int64_t in_channels_, out_channels_, kernel_size_;
};

} // namespace nn
