#pragma once

#include <stdexcept>

#include "../../tensor/tensor.h"
#include "layer.h"
#include "../../ops/dropout_ops.h"

namespace nn {

template <typename T>
class Dropout : public Layer<T> {
private:
    double p_;

public:
    /**
     * Construct inverted dropout with drop probability `p`.
     * Stores `p` after checking it lies in `[0, 1)`.
     *
     * @param p Probability of zeroing an element during training. Default 0.5.
     *
     * @throws std::invalid_argument if `p` is not in `[0, 1)`.
     */
    explicit Dropout(double p = 0.5)
        : p_(p) {
        if (p < 0.0 || p >= 1.0)
            throw std::invalid_argument("Dropout: p must be in [0, 1)");
    }

    /**
     * Apply inverted dropout to `input`.
     * Delegates to `ops::dropout` with this layer's `p_` and `training()` flag.
     * In training, kept values are scaled by `1/(1-p)`; in eval (or `p == 0`) the input is unchanged.
     *
     * @param input Tensor of any shape.
     * @return Dropped (training) or identical (eval) tensor of the same shape.
     */
    tensor::Tensor<T> forward(const tensor::Tensor<T>& input) {
        return ops::dropout(input, p_, this->training());
    }

    /**
     * Return the drop probability.
     *
     * @return `p_`.
     */
    double p() const { return p_; }
};

} // namespace nn
