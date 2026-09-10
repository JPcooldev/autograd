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
    /**
     * Construct an optimizer over an explicit parameter list.
     * Stores `parameters`, `learning_rate`, and `weight_decay` for `step` implementations.
     *
     * @param parameters Non-empty list of parameter tensors to update.
     * @param learning_rate Step size used by derived `step()` implementations.
     * @param weight_decay L2 coefficient (SGD/Adam: coupled in the gradient; AdamW: decoupled). Default 0.
     *
     * @throws std::invalid_argument if `parameters` is empty.
     */
    Optimizer(
        std::vector<tensor::Tensor<T>*> parameters,
        double learning_rate,
        double weight_decay = 0.0
    ) :
        parameters_(parameters),
        learning_rate_(learning_rate),
        weight_decay_(weight_decay) {
        if (parameters_.empty())
            throw std::invalid_argument("Optimizer: parameters cannot be empty");
    }

    /**
     * Destroy the optimizer.
     * Virtual so derived optimizers can be deleted through an Optimizer pointer.
     */
    virtual ~Optimizer() = default;

    /**
     * Apply one parameter update using stored gradients.
     * Implemented by SGD, Adam, and AdamW.
     */
    virtual void step() = 0;

    /**
     * Zero gradients of all managed parameters.
     * Calls `zero_grad()` on each tensor in `parameters_` that has `requires_grad`.
     */
    void zero_grad() {
        for (auto param : parameters_)
            if (param->requires_grad())
                param->zero_grad();
    }
};

} // namespace optim

} // namespace nn
