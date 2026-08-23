#pragma once

#include <stdexcept>
#include <vector>
#include "../../tensor/tensor.h"
#include "../layers/layer.h"

namespace nn {

namespace optim {

template <typename T>
class Optimizer {
protected:
    std::vector<tensor::Tensor<T>*> parameters_;
    double learning_rate_;
    double weight_decay_;
public:
    // Construct from an explicit parameter list.
    Optimizer(
        std::vector<tensor::Tensor<T>*> parameters,
        double learning_rate,
        double weight_decay = 0.0
    ) :
        parameters_(parameters),
        learning_rate_(learning_rate),
        weight_decay_(weight_decay)
    {
        if (parameters_.empty())
            throw std::invalid_argument("Optimizer: parameters cannot be empty");
    }

    virtual ~Optimizer() = default;

    virtual void step() = 0;

    void zero_grad() {
        for (auto param : parameters_) {
            if (param->requires_grad())
                param->zero_grad();
        }
    }
};

} // namespace optim

} // namespace nn