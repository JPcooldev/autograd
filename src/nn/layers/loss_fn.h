#pragma once

#include "../../ops/elementwise_ops.h"
#include "../../ops/loss_ops.h"
#include "../../tensor/tensor.h"
#include "layer.h"

namespace nn {

template <typename T>
class MSELoss : public Layer<T> {
public:
    tensor::Tensor<T> forward(const tensor::Tensor<T>& input, const tensor::Tensor<T>& target) const
    {
        return ops::mse_loss(input, target);
    }
};

template <typename T>
class L1Loss : public Layer<T> {
public:
    tensor::Tensor<T> forward(const tensor::Tensor<T>& input, const tensor::Tensor<T>& target) const
    {
        return ops::l1_loss(input, target);
    }
};

template <typename T>
class CrossEntropyLoss : public Layer<T> {
public:
    tensor::Tensor<T> forward(const tensor::Tensor<T>& input, const tensor::Tensor<T>& target) const
    {
        return ops::cross_entropy_loss(input, target);
    }
};

template <typename T>
class BCELoss : public Layer<T> {
public:
    tensor::Tensor<T> forward(const tensor::Tensor<T>& input, const tensor::Tensor<T>& target) const
    {
        return ops::bce_loss(input, target);
    }
};

template <typename T>
class BCEWithLogitsLoss : public Layer<T> {
public:
    tensor::Tensor<T> forward(const tensor::Tensor<T>& input, const tensor::Tensor<T>& target) const
    {
        return ops::bce_with_logits_loss(input, target);
    }
};

template <typename T>
class KLDivLoss : public Layer<T> {
public:
    tensor::Tensor<T> forward(const tensor::Tensor<T>& input, const tensor::Tensor<T>& target) const
    {
        return ops::kl_div_loss(input, target);
    }
};

} // namespace nn
