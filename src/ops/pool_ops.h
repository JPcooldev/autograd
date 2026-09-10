#pragma once

#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

#include "../tensor/tensor.h"
#include "../autograd/grad_context.h"
#include "../autograd/backward_ops/backward_pool_ops.h"
#include "conv_kernels.h"

namespace ops {

namespace pool_detail {

/**
 * Compute the output shape of a pooling op. Batch and channel sizes are
 * copied from the input; spatial sizes use `conv_out_size` with dilation 1.
 *
 * @param in_shape Input shape `(N, C, *spatial)`.
 * @param kernel Kernel size (same on every spatial axis).
 * @param stride Spatial stride.
 * @param padding Symmetric spatial padding.
 * @return Output shape `(N, C, *out_spatial)`.
 *
 * @throws std::invalid_argument if `stride` is not greater than 0.
 * @throws std::invalid_argument if a computed spatial size is not positive.
 */
inline std::vector<int64_t> pool_out_shape(
    const std::vector<int64_t>& in_shape,
    int64_t kernel, int64_t stride, int64_t padding) {
    const int64_t D = static_cast<int64_t>(in_shape.size()) - 2;
    std::vector<int64_t> out{in_shape[0], in_shape[1]};
    for (int64_t d = 0; d < D; ++d)
        out.push_back(conv_detail::conv_out_size(
            in_shape[static_cast<size_t>(2 + d)], kernel, stride, padding, 1));
    return out;
}

} // namespace pool_detail

/**
 * Apply N-D max pooling (1D, 2D, or 3D spatial). Packs the input, takes the max
 * over each window, and attaches `MaxPoolBackward` with argmax when grad is
 * enabled. Windows that lie entirely in padding write 0.
 *
 * @param input Input tensor of shape `(N, C, *spatial)`.
 * @param kernel Kernel size (same on every spatial axis).
 * @param stride Spatial stride.
 * @param padding Symmetric spatial padding.
 * @return Output tensor of shape `(N, C, *out_spatial)`.
 *
 * @throws std::invalid_argument if `kernel` or `stride` is not greater than 0.
 * @throws std::invalid_argument if the input is not 1D, 2D, or 3D spatial.
 * @throws std::invalid_argument if a computed spatial output size is not positive.
 */
template <typename T>
tensor::Tensor<T> max_pool_nd(
    const tensor::Tensor<T>& input,
    int64_t kernel, int64_t stride, int64_t padding) {
    if (kernel <= 0 || stride <= 0)
        throw std::invalid_argument("max_pool: kernel and stride must be > 0");
    const auto x = input.contiguous();
    const int64_t D = x.rank() - 2;
    if (D < 1 || D > 3)
        throw std::invalid_argument("max_pool: expected 1D, 2D, or 3D input (N, C, *spatial)");

    const auto out_shape = pool_detail::pool_out_shape(x.shape(), kernel, stride, padding);
    const int64_t out_n = tensor::Tensor<T>::compute_numel(out_shape);
    std::vector<T> storage(static_cast<size_t>(out_n));
    std::vector<int64_t> argmax(static_cast<size_t>(out_n), 0);

    using namespace conv_detail;
    const auto in_shape = x.shape();
    const auto in_st = contig_strides(in_shape);
    const auto& xd = x.data();
    std::vector<int64_t> oidx;
    int64_t kvol = 1;
    for (int64_t d = 0; d < D; ++d)
        kvol *= kernel;

    for (int64_t f = 0; f < out_n; ++f) {
        unravel(f, out_shape, oidx);
        const int64_t n = oidx[0];
        const int64_t c = oidx[1];
        T best = std::numeric_limits<T>::lowest();
        int64_t best_i = 0;
        bool any = false;
        for (int64_t kf = 0; kf < kvol; ++kf) {
            int64_t remaining = kf;
            bool inside = true;
            int64_t in_off = n * in_st[0] + c * in_st[1];
            for (int64_t d = D - 1; d >= 0; --d) {
                const int64_t kd = remaining % kernel;
                remaining /= kernel;
                const int64_t in_d = oidx[static_cast<size_t>(2 + d)] * stride - padding + kd;
                if (in_d < 0 || in_d >= in_shape[static_cast<size_t>(2 + d)]) {
                    inside = false;
                    break;
                }
                in_off += in_d * in_st[static_cast<size_t>(2 + d)];
            }
            if (!inside)
                continue;
            if (!any || xd[static_cast<size_t>(in_off)] > best) {
                best = xd[static_cast<size_t>(in_off)];
                best_i = in_off;
                any = true;
            }
        }
        storage[static_cast<size_t>(f)] = any ? best : T{0};
        argmax[static_cast<size_t>(f)] = best_i;
    }

    const bool requires_grad = autograd::is_grad_enabled() && x.requires_grad();
    std::shared_ptr<autograd::Node<T>> grad_fn = nullptr;
    if (requires_grad)
        grad_fn = std::make_shared<autograd::MaxPoolBackward<T>>(x, std::move(argmax));

    return tensor::Tensor<T>::from_operation_result(
        out_shape, std::move(storage), requires_grad, std::move(grad_fn));
}

/**
 * Apply N-D average pooling (1D, 2D, or 3D spatial). Packs the input, averages
 * each window by the full kernel volume (zeros for padding), and attaches
 * `AvgPoolBackward` when grad is enabled.
 *
 * @param input Input tensor of shape `(N, C, *spatial)`.
 * @param kernel Kernel size (same on every spatial axis).
 * @param stride Spatial stride.
 * @param padding Symmetric spatial padding.
 * @return Output tensor of shape `(N, C, *out_spatial)`.
 *
 * @throws std::invalid_argument if `kernel` or `stride` is not greater than 0.
 * @throws std::invalid_argument if the input is not 1D, 2D, or 3D spatial.
 * @throws std::invalid_argument if a computed spatial output size is not positive.
 */
template <typename T>
tensor::Tensor<T> avg_pool_nd(
    const tensor::Tensor<T>& input,
    int64_t kernel, int64_t stride, int64_t padding) {
    if (kernel <= 0 || stride <= 0)
        throw std::invalid_argument("avg_pool: kernel and stride must be > 0");
    const auto x = input.contiguous();
    const int64_t D = x.rank() - 2;
    if (D < 1 || D > 3)
        throw std::invalid_argument("avg_pool: expected 1D, 2D, or 3D input (N, C, *spatial)");

    const auto out_shape = pool_detail::pool_out_shape(x.shape(), kernel, stride, padding);
    const int64_t out_n = tensor::Tensor<T>::compute_numel(out_shape);
    std::vector<T> storage(static_cast<size_t>(out_n));

    using namespace conv_detail;
    const auto in_shape = x.shape();
    const auto in_st = contig_strides(in_shape);
    const auto& xd = x.data();
    int64_t kvol = 1;
    for (int64_t d = 0; d < D; ++d)
        kvol *= kernel;
    const T inv = static_cast<T>(1) / static_cast<T>(kvol);
    std::vector<int64_t> oidx;

    for (int64_t f = 0; f < out_n; ++f) {
        unravel(f, out_shape, oidx);
        const int64_t n = oidx[0];
        const int64_t c = oidx[1];
        T acc = T{0};
        for (int64_t kf = 0; kf < kvol; ++kf) {
            int64_t remaining = kf;
            bool inside = true;
            int64_t in_off = n * in_st[0] + c * in_st[1];
            for (int64_t d = D - 1; d >= 0; --d) {
                const int64_t kd = remaining % kernel;
                remaining /= kernel;
                const int64_t in_d = oidx[static_cast<size_t>(2 + d)] * stride - padding + kd;
                if (in_d < 0 || in_d >= in_shape[static_cast<size_t>(2 + d)]) {
                    inside = false;
                    break;
                }
                in_off += in_d * in_st[static_cast<size_t>(2 + d)];
            }
            if (inside)
                acc += xd[static_cast<size_t>(in_off)];
        }
        storage[static_cast<size_t>(f)] = acc * inv;
    }

    const bool requires_grad = autograd::is_grad_enabled() && x.requires_grad();
    std::shared_ptr<autograd::Node<T>> grad_fn = nullptr;
    if (requires_grad)
        grad_fn = std::make_shared<autograd::AvgPoolBackward<T>>(
            x, kernel, stride, padding, out_shape);

    return tensor::Tensor<T>::from_operation_result(
        out_shape, std::move(storage), requires_grad, std::move(grad_fn));
}

/**
 * Apply 1-D max pooling. Requires input rank 3; default `stride` is `kernel`.
 *
 * @param x Input tensor of shape `(N, C, L)`.
 * @param kernel Kernel size.
 * @param stride Spatial stride, or -1 to use `kernel`.
 * @param padding Symmetric spatial padding.
 * @return Output tensor of shape `(N, C, L_out)`.
 *
 * @throws std::invalid_argument if input rank is not 3.
 * @throws std::invalid_argument if `kernel` or the resolved `stride` is not greater than 0.
 * @throws std::invalid_argument if the computed output length is not positive.
 */
template <typename T>
tensor::Tensor<T> max_pool1d(const tensor::Tensor<T>& x, int64_t kernel, int64_t stride = -1, int64_t padding = 0) {
    if (x.rank() != 3)
        throw std::invalid_argument("max_pool1d: expected (N, C, L)");
    if (stride < 0)
        stride = kernel;
    return max_pool_nd(x, kernel, stride, padding);
}

/**
 * Apply 2-D max pooling. Requires input rank 4; default `stride` is `kernel`.
 *
 * @param x Input tensor of shape `(N, C, H, W)`.
 * @param kernel Kernel size.
 * @param stride Spatial stride, or -1 to use `kernel`.
 * @param padding Symmetric spatial padding.
 * @return Output tensor of shape `(N, C, H_out, W_out)`.
 *
 * @throws std::invalid_argument if input rank is not 4.
 * @throws std::invalid_argument if `kernel` or the resolved `stride` is not greater than 0.
 * @throws std::invalid_argument if a computed spatial output size is not positive.
 */
template <typename T>
tensor::Tensor<T> max_pool2d(const tensor::Tensor<T>& x, int64_t kernel, int64_t stride = -1, int64_t padding = 0) {
    if (x.rank() != 4)
        throw std::invalid_argument("max_pool2d: expected (N, C, H, W)");
    if (stride < 0)
        stride = kernel;
    return max_pool_nd(x, kernel, stride, padding);
}

/**
 * Apply 3-D max pooling. Requires input rank 5; default `stride` is `kernel`.
 *
 * @param x Input tensor of shape `(N, C, D, H, W)`.
 * @param kernel Kernel size.
 * @param stride Spatial stride, or -1 to use `kernel`.
 * @param padding Symmetric spatial padding.
 * @return Output tensor of shape `(N, C, D_out, H_out, W_out)`.
 *
 * @throws std::invalid_argument if input rank is not 5.
 * @throws std::invalid_argument if `kernel` or the resolved `stride` is not greater than 0.
 * @throws std::invalid_argument if a computed spatial output size is not positive.
 */
template <typename T>
tensor::Tensor<T> max_pool3d(const tensor::Tensor<T>& x, int64_t kernel, int64_t stride = -1, int64_t padding = 0) {
    if (x.rank() != 5)
        throw std::invalid_argument("max_pool3d: expected (N, C, D, H, W)");
    if (stride < 0)
        stride = kernel;
    return max_pool_nd(x, kernel, stride, padding);
}

/**
 * Apply 1-D average pooling. Requires input rank 3; default `stride` is `kernel`.
 *
 * @param x Input tensor of shape `(N, C, L)`.
 * @param kernel Kernel size.
 * @param stride Spatial stride, or -1 to use `kernel`.
 * @param padding Symmetric spatial padding.
 * @return Output tensor of shape `(N, C, L_out)`.
 *
 * @throws std::invalid_argument if input rank is not 3.
 * @throws std::invalid_argument if `kernel` or the resolved `stride` is not greater than 0.
 * @throws std::invalid_argument if the computed output length is not positive.
 */
template <typename T>
tensor::Tensor<T> avg_pool1d(const tensor::Tensor<T>& x, int64_t kernel, int64_t stride = -1, int64_t padding = 0) {
    if (x.rank() != 3)
        throw std::invalid_argument("avg_pool1d: expected (N, C, L)");
    if (stride < 0)
        stride = kernel;
    return avg_pool_nd(x, kernel, stride, padding);
}

/**
 * Apply 2-D average pooling. Requires input rank 4; default `stride` is `kernel`.
 *
 * @param x Input tensor of shape `(N, C, H, W)`.
 * @param kernel Kernel size.
 * @param stride Spatial stride, or -1 to use `kernel`.
 * @param padding Symmetric spatial padding.
 * @return Output tensor of shape `(N, C, H_out, W_out)`.
 *
 * @throws std::invalid_argument if input rank is not 4.
 * @throws std::invalid_argument if `kernel` or the resolved `stride` is not greater than 0.
 * @throws std::invalid_argument if a computed spatial output size is not positive.
 */
template <typename T>
tensor::Tensor<T> avg_pool2d(const tensor::Tensor<T>& x, int64_t kernel, int64_t stride = -1, int64_t padding = 0) {
    if (x.rank() != 4)
        throw std::invalid_argument("avg_pool2d: expected (N, C, H, W)");
    if (stride < 0)
        stride = kernel;
    return avg_pool_nd(x, kernel, stride, padding);
}

/**
 * Apply 3-D average pooling. Requires input rank 5; default `stride` is `kernel`.
 *
 * @param x Input tensor of shape `(N, C, D, H, W)`.
 * @param kernel Kernel size.
 * @param stride Spatial stride, or -1 to use `kernel`.
 * @param padding Symmetric spatial padding.
 * @return Output tensor of shape `(N, C, D_out, H_out, W_out)`.
 *
 * @throws std::invalid_argument if input rank is not 5.
 * @throws std::invalid_argument if `kernel` or the resolved `stride` is not greater than 0.
 * @throws std::invalid_argument if a computed spatial output size is not positive.
 */
template <typename T>
tensor::Tensor<T> avg_pool3d(const tensor::Tensor<T>& x, int64_t kernel, int64_t stride = -1, int64_t padding = 0) {
    if (x.rank() != 5)
        throw std::invalid_argument("avg_pool3d: expected (N, C, D, H, W)");
    if (stride < 0)
        stride = kernel;
    return avg_pool_nd(x, kernel, stride, padding);
}

/**
 * Reduce every spatial axis by mean, keeping `(N, C)`. Flattens spatial
 * dimensions then calls `mean` on the last axis.
 *
 * @param input Input tensor of shape `(N, C, *spatial)`.
 * @return Output tensor of shape `(N, C)`.
 *
 * @throws std::invalid_argument if rank is below 3.
 * @throws std::invalid_argument if the flattened spatial dimension is empty.
 */
template <typename T>
tensor::Tensor<T> global_avg_pool(const tensor::Tensor<T>& input) {
    if (input.rank() < 3)
        throw std::invalid_argument("global_avg_pool: expected (N, C, *spatial)");
    auto x = input.flatten(2, -1); // (N, C, spat)
    return x.mean(2);
}

/**
 * Reduce every spatial axis by max, keeping `(N, C)`. Packs, flattens
 * spatial dimensions, then takes the max along the last axis. Attaches
 * `MaxPoolBackward` with argmax when grad is enabled.
 *
 * @param input Input tensor of shape `(N, C, *spatial)`.
 * @return Output tensor of shape `(N, C)`.
 *
 * @throws std::invalid_argument if rank is below 3.
 */
template <typename T>
tensor::Tensor<T> global_max_pool(const tensor::Tensor<T>& input) {
    if (input.rank() < 3)
        throw std::invalid_argument("global_max_pool: expected (N, C, *spatial)");
    const auto x = input.contiguous().flatten(2, -1); // (N, C, spat)
    const int64_t N = x.shape()[0];
    const int64_t C = x.shape()[1];
    const int64_t S = x.shape()[2];
    std::vector<T> storage(static_cast<size_t>(N * C));
    std::vector<int64_t> argmax(static_cast<size_t>(N * C));
    const auto& xd = x.data();
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            const int64_t base = (n * C + c) * S;
            T best = xd[static_cast<size_t>(base)];
            int64_t bi = base;
            for (int64_t s = 1; s < S; ++s) {
                const T v = xd[static_cast<size_t>(base + s)];
                if (v > best) {
                    best = v;
                    bi = base + s;
                }
            }
            storage[static_cast<size_t>(n * C + c)] = best;
            argmax[static_cast<size_t>(n * C + c)] = bi;
        }
    }
    const bool requires_grad = autograd::is_grad_enabled() && x.requires_grad();
    std::shared_ptr<autograd::Node<T>> grad_fn = nullptr;
    if (requires_grad)
        grad_fn = std::make_shared<autograd::MaxPoolBackward<T>>(x, std::move(argmax));
    return tensor::Tensor<T>::from_operation_result(
        {N, C}, std::move(storage), requires_grad, std::move(grad_fn));
}

} // namespace ops
