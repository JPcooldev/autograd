#pragma once

#include <vector>
#include "../../tensor/tensor.h"

namespace nn {

// Abstract base class for layers/modules.
template <typename T>
class Layer {
public:
    virtual ~Layer() = default;

    // Abstract forward method: must be implemented by child layers.
    // Takes an input tensor and returns an output tensor.
    virtual tensor::Tensor<T> forward(const tensor::Tensor<T>& input) const = 0;

    // Override to expose learnable parameters. Leaf layers return their own
    // tensors; Module subclasses recurse into registered sub-modules.
    virtual std::vector<tensor::Tensor<T>*> parameters() { return {}; }
};

} // namespace nn