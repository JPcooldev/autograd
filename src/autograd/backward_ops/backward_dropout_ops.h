#pragma once

#include <cstdint>
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>

#include "../node.h"

namespace autograd {

using tensor::Tensor;

template <typename T>
class DropoutBackward : public Node<T> {
public:
    explicit DropoutBackward(const Tensor<T>& input, const Tensor<T>& mask)
        : Node<T>(input, mask)
    {}

    std::vector<Tensor<T>> apply(const Tensor<T>& propagated_grad) override
    {
        const Tensor<T>& mask = this->saved_tensors[1];
        return {propagated_grad.multiply(mask), Tensor<T>::zeros(mask.shape(), false)};
    }
};

} // namespace autograd
