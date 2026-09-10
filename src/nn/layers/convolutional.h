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

/**
 * Initialize conv `weight` with Kaiming uniform and optional bias.
 * Overwrites `weight` via `kaiming_uniform(a=√5, fan_in)`. If `use_bias`, sets `bias` to
 * U(-1/√fan_in, 1/√fan_in) of shape `{out_channels}`.
 *
 * @param weight Weight tensor to reinitialize (shape already allocated).
 * @param bias Optional bias; assigned only when `use_bias` is true.
 * @param out_channels Length of the bias vector when allocated.
 * @param use_bias Whether to allocate and fill `bias`.
 */
template <typename T>
void kaiming_weight_and_bias(
    tensor::Tensor<T>& weight,
    std::optional<tensor::Tensor<T>>& bias,
    int64_t out_channels,
    bool use_bias) {
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
private:
    int64_t stride_, padding_, dilation_;
    int64_t in_channels_, out_channels_, kernel_size_;

public:
    tensor::Tensor<T> weight;
    std::optional<tensor::Tensor<T>> bias;

    /**
     * Construct a 1-D convolution.
     * Allocates `weight` `{out_channels, in_channels, kernel_size}` then Kaiming-initializes
     * it (and optional bias `{out_channels}`).
     *
     * @param in_channels Number of input channels.
     * @param out_channels Number of output channels.
     * @param kernel_size Convolution kernel length.
     * @param stride Stride. Default 1.
     * @param padding Zeros on both sides of L. Default 0.
     * @param dilation Kernel dilation. Default 1.
     * @param use_bias If true, allocate a learnable bias. Default true.
     */
    Conv1d(int64_t in_channels, int64_t out_channels, int64_t kernel_size,
           int64_t stride = 1, int64_t padding = 0, int64_t dilation = 1,
           bool use_bias = true)
        : stride_(stride), padding_(padding), dilation_(dilation),
          in_channels_(in_channels), out_channels_(out_channels), kernel_size_(kernel_size),
          weight(tensor::Tensor<T>::zeros({out_channels, in_channels, kernel_size}, true)) {
        conv_init::kaiming_weight_and_bias(weight, bias, out_channels, use_bias);
    }

    /**
     * Convolve a rank-3 input `(N, C, L)`.
     * Calls `ops::conv1d` with this layer's weight, optional bias, stride, padding, and dilation.
     *
     * @param input Tensor of shape `(N, C_in, L)`.
     * @return Tensor of shape `(N, C_out, L_out)`.
     *
     * @throws std::invalid_argument if `input` is not rank 3.
     * @throws std::invalid_argument if weight rank or in-channels do not match `input`.
     * @throws std::invalid_argument if bias is present but not shape `{out_channels}`.
     */
    tensor::Tensor<T> forward(const tensor::Tensor<T>& input) const {
        return ops::conv1d(input, weight, bias ? &*bias : nullptr, stride_, padding_, dilation_);
    }

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

template <typename T>
class Conv2d : public Layer<T> {
private:
    int64_t stride_, padding_, dilation_;
    int64_t in_channels_, out_channels_, kernel_size_;

public:
    tensor::Tensor<T> weight;
    std::optional<tensor::Tensor<T>> bias;

    /**
     * Construct a 2-D convolution with a square kernel.
     * Allocates `weight` `{out_channels, in_channels, k, k}` then Kaiming-initializes
     * it (and optional bias `{out_channels}`).
     *
     * @param in_channels Number of input channels.
     * @param out_channels Number of output channels.
     * @param kernel_size Kernel size on H and W.
     * @param stride Stride on H and W. Default 1.
     * @param padding Zeros on both sides of H and W. Default 0.
     * @param dilation Kernel dilation. Default 1.
     * @param use_bias If true, allocate a learnable bias. Default true.
     */
    Conv2d(int64_t in_channels, int64_t out_channels, int64_t kernel_size,
           int64_t stride = 1, int64_t padding = 0, int64_t dilation = 1,
           bool use_bias = true)
        : stride_(stride), padding_(padding), dilation_(dilation),
          in_channels_(in_channels), out_channels_(out_channels), kernel_size_(kernel_size),
          weight(tensor::Tensor<T>::zeros(
              {out_channels, in_channels, kernel_size, kernel_size}, true)) {
        conv_init::kaiming_weight_and_bias(weight, bias, out_channels, use_bias);
    }

    /**
     * Convolve a rank-4 input `(N, C, H, W)`.
     * Calls `ops::conv2d` with this layer's weight, optional bias, stride, padding, and dilation.
     *
     * @param input Tensor of shape `(N, C_in, H, W)`.
     * @return Tensor of shape `(N, C_out, H_out, W_out)`.
     *
     * @throws std::invalid_argument if `input` is not rank 4.
     * @throws std::invalid_argument if weight rank or in-channels do not match `input`.
     * @throws std::invalid_argument if bias is present but not shape `{out_channels}`.
     */
    tensor::Tensor<T> forward(const tensor::Tensor<T>& input) const {
        return ops::conv2d(input, weight, bias ? &*bias : nullptr, stride_, padding_, dilation_);
    }

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

template <typename T>
class Conv3d : public Layer<T> {
private:
    int64_t stride_, padding_, dilation_;
    int64_t in_channels_, out_channels_, kernel_size_;

public:
    tensor::Tensor<T> weight;
    std::optional<tensor::Tensor<T>> bias;

    /**
     * Construct a 3-D convolution with a cubic kernel.
     * Allocates `weight` `{out_channels, in_channels, k, k, k}` then Kaiming-initializes
     * it (and optional bias `{out_channels}`).
     *
     * @param in_channels Number of input channels.
     * @param out_channels Number of output channels.
     * @param kernel_size Kernel size on D, H, and W.
     * @param stride Stride on D, H, and W. Default 1.
     * @param padding Zeros on both sides of each spatial axis. Default 0.
     * @param dilation Kernel dilation. Default 1.
     * @param use_bias If true, allocate a learnable bias. Default true.
     */
    Conv3d(int64_t in_channels, int64_t out_channels, int64_t kernel_size,
           int64_t stride = 1, int64_t padding = 0, int64_t dilation = 1,
           bool use_bias = true)
        : stride_(stride), padding_(padding), dilation_(dilation),
          in_channels_(in_channels), out_channels_(out_channels), kernel_size_(kernel_size),
          weight(tensor::Tensor<T>::zeros(
              {out_channels, in_channels, kernel_size, kernel_size, kernel_size}, true)) {
        conv_init::kaiming_weight_and_bias(weight, bias, out_channels, use_bias);
    }

    /**
     * Convolve a rank-5 input `(N, C, D, H, W)`.
     * Calls `ops::conv3d` with this layer's weight, optional bias, stride, padding, and dilation.
     *
     * @param input Tensor of shape `(N, C_in, D, H, W)`.
     * @return Tensor of shape `(N, C_out, D_out, H_out, W_out)`.
     *
     * @throws std::invalid_argument if `input` is not rank 5.
     * @throws std::invalid_argument if weight rank or in-channels do not match `input`.
     * @throws std::invalid_argument if bias is present but not shape `{out_channels}`.
     */
    tensor::Tensor<T> forward(const tensor::Tensor<T>& input) const {
        return ops::conv3d(input, weight, bias ? &*bias : nullptr, stride_, padding_, dilation_);
    }

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

// ---------------------------------------------------------------------------
// ConvTranspose1d / 2d / 3d
// ---------------------------------------------------------------------------
// W shape {in_channels, out_channels, *kernel}

template <typename T>
class ConvTranspose1d : public Layer<T> {
private:
    int64_t stride_, padding_, dilation_;
    int64_t in_channels_, out_channels_, kernel_size_;

public:
    tensor::Tensor<T> weight;
    std::optional<tensor::Tensor<T>> bias;

    /**
     * Construct a 1-D transposed convolution.
     * Allocates `weight` `{in_channels, out_channels, kernel_size}` then Kaiming-initializes
     * it (and optional bias `{out_channels}`). `output_padding` is 0 in `forward`.
     *
     * @param in_channels Number of input channels.
     * @param out_channels Number of output channels.
     * @param kernel_size Convolution kernel length.
     * @param stride Stride. Default 1.
     * @param padding Cropping on both sides of L. Default 0.
     * @param dilation Kernel dilation. Default 1.
     * @param use_bias If true, allocate a learnable bias. Default true.
     */
    ConvTranspose1d(int64_t in_channels, int64_t out_channels, int64_t kernel_size,
                    int64_t stride = 1, int64_t padding = 0, int64_t dilation = 1,
                    bool use_bias = true)
        : stride_(stride), padding_(padding), dilation_(dilation),
          in_channels_(in_channels), out_channels_(out_channels), kernel_size_(kernel_size),
          weight(tensor::Tensor<T>::zeros({in_channels, out_channels, kernel_size}, true)) {
        conv_init::kaiming_weight_and_bias(weight, bias, out_channels, use_bias);
    }

    /**
     * Apply 1-D transposed convolution to `(N, C, L)`.
     * Calls `ops::conv_transpose1d` with `output_padding = 0`.
     *
     * @param input Tensor of shape `(N, C_in, L)`.
     * @return Tensor of shape `(N, C_out, L_out)`.
     *
     * @throws std::invalid_argument if `input` is not rank 3.
     * @throws std::invalid_argument if weight rank or in-channels do not match `input`.
     * @throws std::invalid_argument if bias is present but not shape `{out_channels}`.
     */
    tensor::Tensor<T> forward(const tensor::Tensor<T>& input) const {
        return ops::conv_transpose1d(
            input, weight, bias ? &*bias : nullptr, stride_, padding_, dilation_);
    }

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

template <typename T>
class ConvTranspose2d : public Layer<T> {
private:
    int64_t stride_, padding_, dilation_;
    int64_t in_channels_, out_channels_, kernel_size_;

public:
    tensor::Tensor<T> weight;
    std::optional<tensor::Tensor<T>> bias;

    /**
     * Construct a 2-D transposed convolution with a square kernel.
     * Allocates `weight` `{in_channels, out_channels, k, k}` then Kaiming-initializes
     * it (and optional bias `{out_channels}`). `output_padding` is 0 in `forward`.
     *
     * @param in_channels Number of input channels.
     * @param out_channels Number of output channels.
     * @param kernel_size Kernel size on H and W.
     * @param stride Stride on H and W. Default 1.
     * @param padding Cropping on both sides of H and W. Default 0.
     * @param dilation Kernel dilation. Default 1.
     * @param use_bias If true, allocate a learnable bias. Default true.
     */
    ConvTranspose2d(int64_t in_channels, int64_t out_channels, int64_t kernel_size,
                    int64_t stride = 1, int64_t padding = 0, int64_t dilation = 1,
                    bool use_bias = true)
        : stride_(stride), padding_(padding), dilation_(dilation),
          in_channels_(in_channels), out_channels_(out_channels), kernel_size_(kernel_size),
          weight(tensor::Tensor<T>::zeros(
              {in_channels, out_channels, kernel_size, kernel_size}, true)) {
        conv_init::kaiming_weight_and_bias(weight, bias, out_channels, use_bias);
    }

    /**
     * Apply 2-D transposed convolution to `(N, C, H, W)`.
     * Calls `ops::conv_transpose2d` with `output_padding = 0`.
     *
     * @param input Tensor of shape `(N, C_in, H, W)`.
     * @return Tensor of shape `(N, C_out, H_out, W_out)`.
     *
     * @throws std::invalid_argument if `input` is not rank 4.
     * @throws std::invalid_argument if weight rank or in-channels do not match `input`.
     * @throws std::invalid_argument if bias is present but not shape `{out_channels}`.
     */
    tensor::Tensor<T> forward(const tensor::Tensor<T>& input) const {
        return ops::conv_transpose2d(
            input, weight, bias ? &*bias : nullptr, stride_, padding_, dilation_);
    }

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

template <typename T>
class ConvTranspose3d : public Layer<T> {
private:
    int64_t stride_, padding_, dilation_;
    int64_t in_channels_, out_channels_, kernel_size_;

public:
    tensor::Tensor<T> weight;
    std::optional<tensor::Tensor<T>> bias;

    /**
     * Construct a 3-D transposed convolution with a cubic kernel.
     * Allocates `weight` `{in_channels, out_channels, k, k, k}` then Kaiming-initializes
     * it (and optional bias `{out_channels}`). `output_padding` is 0 in `forward`.
     *
     * @param in_channels Number of input channels.
     * @param out_channels Number of output channels.
     * @param kernel_size Kernel size on D, H, and W.
     * @param stride Stride on D, H, and W. Default 1.
     * @param padding Cropping on both sides of each spatial axis. Default 0.
     * @param dilation Kernel dilation. Default 1.
     * @param use_bias If true, allocate a learnable bias. Default true.
     */
    ConvTranspose3d(int64_t in_channels, int64_t out_channels, int64_t kernel_size,
                    int64_t stride = 1, int64_t padding = 0, int64_t dilation = 1,
                    bool use_bias = true)
        : stride_(stride), padding_(padding), dilation_(dilation),
          in_channels_(in_channels), out_channels_(out_channels), kernel_size_(kernel_size),
          weight(tensor::Tensor<T>::zeros(
              {in_channels, out_channels, kernel_size, kernel_size, kernel_size}, true)) {
        conv_init::kaiming_weight_and_bias(weight, bias, out_channels, use_bias);
    }

    /**
     * Apply 3-D transposed convolution to `(N, C, D, H, W)`.
     * Calls `ops::conv_transpose3d` with `output_padding = 0`.
     *
     * @param input Tensor of shape `(N, C_in, D, H, W)`.
     * @return Tensor of shape `(N, C_out, D_out, H_out, W_out)`.
     *
     * @throws std::invalid_argument if `input` is not rank 5.
     * @throws std::invalid_argument if weight rank or in-channels do not match `input`.
     * @throws std::invalid_argument if bias is present but not shape `{out_channels}`.
     */
    tensor::Tensor<T> forward(const tensor::Tensor<T>& input) const {
        return ops::conv_transpose3d(
            input, weight, bias ? &*bias : nullptr, stride_, padding_, dilation_);
    }

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
