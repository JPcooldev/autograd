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
    /**
     * Construct the backward node for dropout.
     * Saves aliases of `input` and the keep `mask` via the binary `Node` ctor.
     *
     * @param input Forward input.
     * @param mask Keep-mask used in the forward pass (same shape as `input`).
     */
    explicit DropoutBackward(const Tensor<T>& input, const Tensor<T>& mask)
        : Node<T>(input, mask) {}

    /**
     * Compute the dropout input gradient and a zero gradient for the mask.
     * Multiplies `propagated_grad` by saved `mask` (`saved_tensors[1]`);
     * `saved_tensors[0]` (input) is unused.
     *
     * @param propagated_grad Upstream gradient dL/d(dropout output).
     * @return `{dL/d(input), zeros_like(mask)}`.
     *
     * @throws std::invalid_argument if `propagated_grad` and the mask have
     *         different shapes.
     */
    std::vector<Tensor<T>> apply(const Tensor<T>& propagated_grad) override {
        const Tensor<T>& mask = this->saved_tensors[1];
        return {propagated_grad.multiply(mask), Tensor<T>::zeros(mask.shape(), false)};
    }
};

} // namespace autograd
