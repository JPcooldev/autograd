#pragma once

#include <stdexcept>

#include "../../tensor/tensor.h"
#include "layer.h"
#include "../../ops/dropout_ops.h"

namespace nn {

template <typename T>
class Dropout : public Layer<T> {
public:
    explicit Dropout(double p = 0.5)
        : p_(p)
    {
        if (p < 0.0 || p >= 1.0)
            throw std::invalid_argument("Dropout: p must be in [0, 1)");
    }

    tensor::Tensor<T> forward(const tensor::Tensor<T>& input)
    {
        return ops::dropout(input, p_, this->training());
    }

    double p() const { return p_; }

private:
    double p_;
};

} // namespace nn
