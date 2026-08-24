#pragma once

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

#include "../tensor/tensor.h"
#include "../autograd/grad_context.h"
#include "../autograd/backward_ops/backward_conv_ops.h"
#include "conv_kernels.h"

namespace ops {

namespace conv_detail_wrap {

template <typename T>
std::vector<int64_t> conv_out_shape(
    const std::vector<int64_t>& in_shape,
    const std::vector<int64_t>& w_shape,
    int64_t stride, int64_t padding, int64_t dilation)
{
    const int64_t D = static_cast<int64_t>(in_shape.size()) - 2;
    std::vector<int64_t> out{in_shape[0], w_shape[0]};
    out.reserve(static_cast<size_t>(2 + D));
    for (int64_t d = 0; d < D; ++d) {
        out.push_back(conv_detail::conv_out_size(
            in_shape[static_cast<size_t>(2 + d)],
            w_shape[static_cast<size_t>(2 + d)],
            stride, padding, dilation));
    }
    return out;
}

template <typename T>
std::vector<int64_t> conv_transpose_out_shape(
    const std::vector<int64_t>& in_shape,
    const std::vector<int64_t>& w_shape,
    int64_t stride, int64_t padding, int64_t dilation, int64_t output_padding)
{
    const int64_t D = static_cast<int64_t>(in_shape.size()) - 2;
    std::vector<int64_t> out{in_shape[0], w_shape[1]};
    for (int64_t d = 0; d < D; ++d) {
        out.push_back(conv_detail::conv_transpose_out_size(
            in_shape[static_cast<size_t>(2 + d)],
            w_shape[static_cast<size_t>(2 + d)],
            stride, padding, dilation, output_padding));
    }
    return out;
}

} // namespace conv_detail_wrap

template <typename T>
tensor::Tensor<T> conv_nd(
    const tensor::Tensor<T>& input,
    const tensor::Tensor<T>& weight,
    const tensor::Tensor<T>* bias,
    int64_t stride,
    int64_t padding,
    int64_t dilation)
{
    const auto x = input.contiguous();
    const auto w = weight.contiguous();
    const int64_t D = x.rank() - 2;
    if (D < 1 || D > 3)
        throw std::invalid_argument("conv: expected 1D, 2D, or 3D input (N, C, *spatial)");
    if (w.rank() != x.rank())
        throw std::invalid_argument("conv: weight rank must match input rank");
    if (w.shape()[1] != x.shape()[1])
        throw std::invalid_argument("conv: in_channels mismatch");
    if (bias && (bias->rank() != 1 || bias->shape()[0] != w.shape()[0]))
        throw std::invalid_argument("conv: bias must have shape {out_channels}");

    const auto out_shape = conv_detail_wrap::conv_out_shape<T>(
        x.shape(), w.shape(), stride, padding, dilation);
    std::vector<T> storage(static_cast<size_t>(tensor::Tensor<T>::compute_numel(out_shape)));
    const T* bptr = nullptr;
    tensor::Tensor<T> bias_c({});
    if (bias) {
        bias_c = bias->contiguous();
        bptr = bias_c.data().data();
    }
    conv_detail::conv_forward(
        x.data().data(), x.shape(),
        w.data().data(), w.shape(),
        bptr,
        storage.data(), out_shape,
        stride, padding, dilation);

    const bool requires_grad = autograd::is_grad_enabled() &&
        (x.requires_grad() || w.requires_grad() || (bias && bias->requires_grad()));
    std::shared_ptr<autograd::Node<T>> grad_fn = nullptr;
    if (requires_grad)
        grad_fn = std::make_shared<autograd::ConvBackward<T>>(
            x, w, bias, stride, padding, dilation, out_shape);

    return tensor::Tensor<T>::from_operation_result(
        out_shape, std::move(storage), requires_grad, std::move(grad_fn));
}

template <typename T>
tensor::Tensor<T> conv_transpose_nd(
    const tensor::Tensor<T>& input,
    const tensor::Tensor<T>& weight,
    const tensor::Tensor<T>* bias,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    int64_t output_padding)
{
    const auto x = input.contiguous();
    const auto w = weight.contiguous();
    const int64_t D = x.rank() - 2;
    if (D < 1 || D > 3)
        throw std::invalid_argument("conv_transpose: expected 1D, 2D, or 3D input");
    if (w.rank() != x.rank())
        throw std::invalid_argument("conv_transpose: weight rank must match input rank");
    if (w.shape()[0] != x.shape()[1])
        throw std::invalid_argument("conv_transpose: in_channels mismatch");
    if (bias && (bias->rank() != 1 || bias->shape()[0] != w.shape()[1]))
        throw std::invalid_argument("conv_transpose: bias must have shape {out_channels}");

    const auto out_shape = conv_detail_wrap::conv_transpose_out_shape<T>(
        x.shape(), w.shape(), stride, padding, dilation, output_padding);
    std::vector<T> storage(static_cast<size_t>(tensor::Tensor<T>::compute_numel(out_shape)));
    const T* bptr = nullptr;
    tensor::Tensor<T> bias_c({});
    if (bias) {
        bias_c = bias->contiguous();
        bptr = bias_c.data().data();
    }
    conv_detail::conv_transpose_forward(
        x.data().data(), x.shape(),
        w.data().data(), w.shape(),
        bptr,
        storage.data(), out_shape,
        stride, padding, dilation);

    const bool requires_grad = autograd::is_grad_enabled() &&
        (x.requires_grad() || w.requires_grad() || (bias && bias->requires_grad()));
    std::shared_ptr<autograd::Node<T>> grad_fn = nullptr;
    if (requires_grad)
        grad_fn = std::make_shared<autograd::ConvTransposeBackward<T>>(
            x, w, bias, stride, padding, dilation, out_shape);

    return tensor::Tensor<T>::from_operation_result(
        out_shape, std::move(storage), requires_grad, std::move(grad_fn));
}

template <typename T>
tensor::Tensor<T> conv1d(const tensor::Tensor<T>& input, const tensor::Tensor<T>& weight,
                         const tensor::Tensor<T>* bias = nullptr,
                         int64_t stride = 1, int64_t padding = 0, int64_t dilation = 1)
{
    if (input.rank() != 3)
        throw std::invalid_argument("conv1d: input must have shape (N, C, L)");
    return conv_nd(input, weight, bias, stride, padding, dilation);
}

template <typename T>
tensor::Tensor<T> conv2d(const tensor::Tensor<T>& input, const tensor::Tensor<T>& weight,
                         const tensor::Tensor<T>* bias = nullptr,
                         int64_t stride = 1, int64_t padding = 0, int64_t dilation = 1)
{
    if (input.rank() != 4)
        throw std::invalid_argument("conv2d: input must have shape (N, C, H, W)");
    return conv_nd(input, weight, bias, stride, padding, dilation);
}

template <typename T>
tensor::Tensor<T> conv3d(const tensor::Tensor<T>& input, const tensor::Tensor<T>& weight,
                         const tensor::Tensor<T>* bias = nullptr,
                         int64_t stride = 1, int64_t padding = 0, int64_t dilation = 1)
{
    if (input.rank() != 5)
        throw std::invalid_argument("conv3d: input must have shape (N, C, D, H, W)");
    return conv_nd(input, weight, bias, stride, padding, dilation);
}

template <typename T>
tensor::Tensor<T> conv_transpose1d(const tensor::Tensor<T>& input, const tensor::Tensor<T>& weight,
                                   const tensor::Tensor<T>* bias = nullptr,
                                   int64_t stride = 1, int64_t padding = 0,
                                   int64_t dilation = 1, int64_t output_padding = 0)
{
    if (input.rank() != 3)
        throw std::invalid_argument("conv_transpose1d: input must have shape (N, C, L)");
    return conv_transpose_nd(input, weight, bias, stride, padding, dilation, output_padding);
}

template <typename T>
tensor::Tensor<T> conv_transpose2d(const tensor::Tensor<T>& input, const tensor::Tensor<T>& weight,
                                   const tensor::Tensor<T>* bias = nullptr,
                                   int64_t stride = 1, int64_t padding = 0,
                                   int64_t dilation = 1, int64_t output_padding = 0)
{
    if (input.rank() != 4)
        throw std::invalid_argument("conv_transpose2d: input must have shape (N, C, H, W)");
    return conv_transpose_nd(input, weight, bias, stride, padding, dilation, output_padding);
}

template <typename T>
tensor::Tensor<T> conv_transpose3d(const tensor::Tensor<T>& input, const tensor::Tensor<T>& weight,
                                   const tensor::Tensor<T>* bias = nullptr,
                                   int64_t stride = 1, int64_t padding = 0,
                                   int64_t dilation = 1, int64_t output_padding = 0)
{
    if (input.rank() != 5)
        throw std::invalid_argument("conv_transpose3d: input must have shape (N, C, D, H, W)");
    return conv_transpose_nd(input, weight, bias, stride, padding, dilation, output_padding);
}

} // namespace ops
