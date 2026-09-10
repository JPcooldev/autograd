#pragma once

#include <cstdint>
#include <stdexcept>
#include <vector>

#include "../../tensor/tensor.h"
#include "layer.h"
#include "../../ops/pool_ops.h"

namespace nn {

template <typename T>
class MaxPool1d : public Layer<T> {
private:
    int64_t kernel_size_, stride_, padding_;

public:
    /**
     * Construct 1-D max pooling.
     * If `stride < 0`, stride is set to `kernel_size` (non-overlapping windows).
     *
     * @param kernel_size Window length.
     * @param stride Step between windows. Default `-1` means `kernel_size`.
     * @param padding Zeros added on both sides of the length axis. Default 0.
     */
    MaxPool1d(int64_t kernel_size, int64_t stride = -1, int64_t padding = 0)
        : kernel_size_(kernel_size), stride_(stride < 0 ? kernel_size : stride), padding_(padding) {}

    /**
     * Max-pool a rank-3 input `(N, C, L)`.
     * Calls `ops::max_pool1d` with stored kernel, stride, and padding.
     *
     * @param input Tensor of shape `(N, C, L)`.
     * @return Pooled tensor `(N, C, L_out)`.
     *
     * @throws std::invalid_argument if `input` is not rank 3.
     * @throws std::invalid_argument if `kernel_size` or `stride` is not greater than 0.
     */
    tensor::Tensor<T> forward(const tensor::Tensor<T>& input) const {
        return ops::max_pool1d(input, kernel_size_, stride_, padding_);
    }
};

template <typename T>
class MaxPool2d : public Layer<T> {
private:
    int64_t kernel_size_, stride_, padding_;

public:
    /**
     * Construct 2-D max pooling with a square window.
     * If `stride < 0`, stride is set to `kernel_size`.
     *
     * @param kernel_size Window size on H and W.
     * @param stride Step on H and W. Default `-1` means `kernel_size`.
     * @param padding Zeros on both sides of H and W. Default 0.
     */
    MaxPool2d(int64_t kernel_size, int64_t stride = -1, int64_t padding = 0)
        : kernel_size_(kernel_size), stride_(stride < 0 ? kernel_size : stride), padding_(padding) {}

    /**
     * Max-pool a rank-4 input `(N, C, H, W)`.
     * Calls `ops::max_pool2d` with stored kernel, stride, and padding.
     *
     * @param input Tensor of shape `(N, C, H, W)`.
     * @return Pooled tensor `(N, C, H_out, W_out)`.
     *
     * @throws std::invalid_argument if `input` is not rank 4.
     * @throws std::invalid_argument if `kernel_size` or `stride` is not greater than 0.
     */
    tensor::Tensor<T> forward(const tensor::Tensor<T>& input) const {
        return ops::max_pool2d(input, kernel_size_, stride_, padding_);
    }
};

template <typename T>
class MaxPool3d : public Layer<T> {
private:
    int64_t kernel_size_, stride_, padding_;

public:
    /**
     * Construct 3-D max pooling with a cubic window.
     * If `stride < 0`, stride is set to `kernel_size`.
     *
     * @param kernel_size Window size on D, H, and W.
     * @param stride Step on D, H, and W. Default `-1` means `kernel_size`.
     * @param padding Zeros on both sides of each spatial axis. Default 0.
     */
    MaxPool3d(int64_t kernel_size, int64_t stride = -1, int64_t padding = 0)
        : kernel_size_(kernel_size), stride_(stride < 0 ? kernel_size : stride), padding_(padding) {}

    /**
     * Max-pool a rank-5 input `(N, C, D, H, W)`.
     * Calls `ops::max_pool3d` with stored kernel, stride, and padding.
     *
     * @param input Tensor of shape `(N, C, D, H, W)`.
     * @return Pooled tensor `(N, C, D_out, H_out, W_out)`.
     *
     * @throws std::invalid_argument if `input` is not rank 5.
     * @throws std::invalid_argument if `kernel_size` or `stride` is not greater than 0.
     */
    tensor::Tensor<T> forward(const tensor::Tensor<T>& input) const {
        return ops::max_pool3d(input, kernel_size_, stride_, padding_);
    }
};

template <typename T>
class AvgPool1d : public Layer<T> {
private:
    int64_t kernel_size_, stride_, padding_;

public:
    /**
     * Construct 1-D average pooling.
     * If `stride < 0`, stride is set to `kernel_size`.
     *
     * @param kernel_size Window length.
     * @param stride Step between windows. Default `-1` means `kernel_size`.
     * @param padding Zeros added on both sides of the length axis. Default 0.
     */
    AvgPool1d(int64_t kernel_size, int64_t stride = -1, int64_t padding = 0)
        : kernel_size_(kernel_size), stride_(stride < 0 ? kernel_size : stride), padding_(padding) {}

    /**
     * Average-pool a rank-3 input `(N, C, L)`.
     * Calls `ops::avg_pool1d` (`count_include_pad` is true in the op: divide by `k`, including zeros).
     *
     * @param input Tensor of shape `(N, C, L)`.
     * @return Pooled tensor `(N, C, L_out)`.
     *
     * @throws std::invalid_argument if `input` is not rank 3.
     * @throws std::invalid_argument if `kernel_size` or `stride` is not greater than 0.
     */
    tensor::Tensor<T> forward(const tensor::Tensor<T>& input) const {
        return ops::avg_pool1d(input, kernel_size_, stride_, padding_);
    }
};

template <typename T>
class AvgPool2d : public Layer<T> {
private:
    int64_t kernel_size_, stride_, padding_;

public:
    /**
     * Construct 2-D average pooling with a square window.
     * If `stride < 0`, stride is set to `kernel_size`.
     *
     * @param kernel_size Window size on H and W.
     * @param stride Step on H and W. Default `-1` means `kernel_size`.
     * @param padding Zeros on both sides of H and W. Default 0.
     */
    AvgPool2d(int64_t kernel_size, int64_t stride = -1, int64_t padding = 0)
        : kernel_size_(kernel_size), stride_(stride < 0 ? kernel_size : stride), padding_(padding) {}

    /**
     * Average-pool a rank-4 input `(N, C, H, W)`.
     * Calls `ops::avg_pool2d` with stored kernel, stride, and padding.
     *
     * @param input Tensor of shape `(N, C, H, W)`.
     * @return Pooled tensor `(N, C, H_out, W_out)`.
     *
     * @throws std::invalid_argument if `input` is not rank 4.
     * @throws std::invalid_argument if `kernel_size` or `stride` is not greater than 0.
     */
    tensor::Tensor<T> forward(const tensor::Tensor<T>& input) const {
        return ops::avg_pool2d(input, kernel_size_, stride_, padding_);
    }
};

template <typename T>
class AvgPool3d : public Layer<T> {
private:
    int64_t kernel_size_, stride_, padding_;

public:
    /**
     * Construct 3-D average pooling with a cubic window.
     * If `stride < 0`, stride is set to `kernel_size`.
     *
     * @param kernel_size Window size on D, H, and W.
     * @param stride Step on D, H, and W. Default `-1` means `kernel_size`.
     * @param padding Zeros on both sides of each spatial axis. Default 0.
     */
    AvgPool3d(int64_t kernel_size, int64_t stride = -1, int64_t padding = 0)
        : kernel_size_(kernel_size), stride_(stride < 0 ? kernel_size : stride), padding_(padding) {}

    /**
     * Average-pool a rank-5 input `(N, C, D, H, W)`.
     * Calls `ops::avg_pool3d` with stored kernel, stride, and padding.
     *
     * @param input Tensor of shape `(N, C, D, H, W)`.
     * @return Pooled tensor `(N, C, D_out, H_out, W_out)`.
     *
     * @throws std::invalid_argument if `input` is not rank 5.
     * @throws std::invalid_argument if `kernel_size` or `stride` is not greater than 0.
     */
    tensor::Tensor<T> forward(const tensor::Tensor<T>& input) const {
        return ops::avg_pool3d(input, kernel_size_, stride_, padding_);
    }
};

template <typename T>
class GlobalMaxPool1d : public Layer<T> {
public:
    /**
     * Reduce every spatial axis with max, keeping `(N, C)`.
     * Calls `ops::global_max_pool` (rank must be ≥ 3).
     *
     * @param input Tensor of shape `(N, C, *spatial)`.
     * @return Tensor of shape `(N, C)`.
     *
     * @throws std::invalid_argument if `input.rank() < 3`.
     */
    tensor::Tensor<T> forward(const tensor::Tensor<T>& input) const {
        return ops::global_max_pool(input);
    }
};

template <typename T>
class GlobalMaxPool2d : public Layer<T> {
public:
    /**
     * Reduce every spatial axis with max, keeping `(N, C)`.
     * Calls `ops::global_max_pool` (rank must be ≥ 3).
     *
     * @param input Tensor of shape `(N, C, *spatial)`.
     * @return Tensor of shape `(N, C)`.
     *
     * @throws std::invalid_argument if `input.rank() < 3`.
     */
    tensor::Tensor<T> forward(const tensor::Tensor<T>& input) const {
        return ops::global_max_pool(input);
    }
};

template <typename T>
class GlobalMaxPool3d : public Layer<T> {
public:
    /**
     * Reduce every spatial axis with max, keeping `(N, C)`.
     * Calls `ops::global_max_pool` (rank must be ≥ 3).
     *
     * @param input Tensor of shape `(N, C, *spatial)`.
     * @return Tensor of shape `(N, C)`.
     *
     * @throws std::invalid_argument if `input.rank() < 3`.
     */
    tensor::Tensor<T> forward(const tensor::Tensor<T>& input) const {
        return ops::global_max_pool(input);
    }
};

template <typename T>
class GlobalAvgPool1d : public Layer<T> {
public:
    /**
     * Reduce every spatial axis with mean, keeping `(N, C)`.
     * Calls `ops::global_avg_pool` (rank must be ≥ 3).
     *
     * @param input Tensor of shape `(N, C, *spatial)`.
     * @return Tensor of shape `(N, C)`.
     *
     * @throws std::invalid_argument if `input.rank() < 3`.
     */
    tensor::Tensor<T> forward(const tensor::Tensor<T>& input) const {
        return ops::global_avg_pool(input);
    }
};

template <typename T>
class GlobalAvgPool2d : public Layer<T> {
public:
    /**
     * Reduce every spatial axis with mean, keeping `(N, C)`.
     * Calls `ops::global_avg_pool` (rank must be ≥ 3).
     *
     * @param input Tensor of shape `(N, C, *spatial)`.
     * @return Tensor of shape `(N, C)`.
     *
     * @throws std::invalid_argument if `input.rank() < 3`.
     */
    tensor::Tensor<T> forward(const tensor::Tensor<T>& input) const {
        return ops::global_avg_pool(input);
    }
};

template <typename T>
class GlobalAvgPool3d : public Layer<T> {
public:
    /**
     * Reduce every spatial axis with mean, keeping `(N, C)`.
     * Calls `ops::global_avg_pool` (rank must be ≥ 3).
     *
     * @param input Tensor of shape `(N, C, *spatial)`.
     * @return Tensor of shape `(N, C)`.
     *
     * @throws std::invalid_argument if `input.rank() < 3`.
     */
    tensor::Tensor<T> forward(const tensor::Tensor<T>& input) const {
        return ops::global_avg_pool(input);
    }
};

} // namespace nn
