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
public:
    MaxPool1d(int64_t kernel_size, int64_t stride = -1, int64_t padding = 0)
        : kernel_size_(kernel_size), stride_(stride < 0 ? kernel_size : stride), padding_(padding)
    {}
    tensor::Tensor<T> forward(const tensor::Tensor<T>& input) const
    {
        return ops::max_pool1d(input, kernel_size_, stride_, padding_);
    }
private:
    int64_t kernel_size_, stride_, padding_;
};

template <typename T>
class MaxPool2d : public Layer<T> {
public:
    MaxPool2d(int64_t kernel_size, int64_t stride = -1, int64_t padding = 0)
        : kernel_size_(kernel_size), stride_(stride < 0 ? kernel_size : stride), padding_(padding)
    {}
    tensor::Tensor<T> forward(const tensor::Tensor<T>& input) const
    {
        return ops::max_pool2d(input, kernel_size_, stride_, padding_);
    }
private:
    int64_t kernel_size_, stride_, padding_;
};

template <typename T>
class MaxPool3d : public Layer<T> {
public:
    MaxPool3d(int64_t kernel_size, int64_t stride = -1, int64_t padding = 0)
        : kernel_size_(kernel_size), stride_(stride < 0 ? kernel_size : stride), padding_(padding)
    {}
    tensor::Tensor<T> forward(const tensor::Tensor<T>& input) const
    {
        return ops::max_pool3d(input, kernel_size_, stride_, padding_);
    }
private:
    int64_t kernel_size_, stride_, padding_;
};

template <typename T>
class AvgPool1d : public Layer<T> {
public:
    AvgPool1d(int64_t kernel_size, int64_t stride = -1, int64_t padding = 0)
        : kernel_size_(kernel_size), stride_(stride < 0 ? kernel_size : stride), padding_(padding)
    {}
    tensor::Tensor<T> forward(const tensor::Tensor<T>& input) const
    {
        return ops::avg_pool1d(input, kernel_size_, stride_, padding_);
    }
private:
    int64_t kernel_size_, stride_, padding_;
};

template <typename T>
class AvgPool2d : public Layer<T> {
public:
    AvgPool2d(int64_t kernel_size, int64_t stride = -1, int64_t padding = 0)
        : kernel_size_(kernel_size), stride_(stride < 0 ? kernel_size : stride), padding_(padding)
    {}
    tensor::Tensor<T> forward(const tensor::Tensor<T>& input) const
    {
        return ops::avg_pool2d(input, kernel_size_, stride_, padding_);
    }
private:
    int64_t kernel_size_, stride_, padding_;
};

template <typename T>
class AvgPool3d : public Layer<T> {
public:
    AvgPool3d(int64_t kernel_size, int64_t stride = -1, int64_t padding = 0)
        : kernel_size_(kernel_size), stride_(stride < 0 ? kernel_size : stride), padding_(padding)
    {}
    tensor::Tensor<T> forward(const tensor::Tensor<T>& input) const
    {
        return ops::avg_pool3d(input, kernel_size_, stride_, padding_);
    }
private:
    int64_t kernel_size_, stride_, padding_;
};

template <typename T>
class GlobalMaxPool1d : public Layer<T> {
public:
    tensor::Tensor<T> forward(const tensor::Tensor<T>& input) const
    {
        return ops::global_max_pool(input);
    }
};
template <typename T>
class GlobalMaxPool2d : public Layer<T> {
public:
    tensor::Tensor<T> forward(const tensor::Tensor<T>& input) const
    {
        return ops::global_max_pool(input);
    }
};
template <typename T>
class GlobalMaxPool3d : public Layer<T> {
public:
    tensor::Tensor<T> forward(const tensor::Tensor<T>& input) const
    {
        return ops::global_max_pool(input);
    }
};
template <typename T>
class GlobalAvgPool1d : public Layer<T> {
public:
    tensor::Tensor<T> forward(const tensor::Tensor<T>& input) const
    {
        return ops::global_avg_pool(input);
    }
};
template <typename T>
class GlobalAvgPool2d : public Layer<T> {
public:
    tensor::Tensor<T> forward(const tensor::Tensor<T>& input) const
    {
        return ops::global_avg_pool(input);
    }
};
template <typename T>
class GlobalAvgPool3d : public Layer<T> {
public:
    tensor::Tensor<T> forward(const tensor::Tensor<T>& input) const
    {
        return ops::global_avg_pool(input);
    }
};

} // namespace nn
