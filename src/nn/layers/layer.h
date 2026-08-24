#pragma once

#include <vector>
#include "../../tensor/tensor.h"

namespace nn {

// Abstract base class for layers/modules.
//
// Each concrete layer defines its own `forward` signature (Python nn.Module
// does the same). There is no virtual single-tensor forward — losses take
// (input, target) and RNNs return (output, hidden).
template <typename T>
class Layer {
public:
    virtual ~Layer() = default;

    // PyTorch-like train/eval. `train()` is virtual so Module can recurse.
    virtual void train(bool mode = true) { training_ = mode; }
    void eval() { train(false); }
    bool training() const { return training_; }

    // Override to expose learnable parameters. Leaf layers return their own
    // tensors; Module subclasses recurse into registered sub-modules.
    virtual std::vector<tensor::Tensor<T>*> parameters() { return {}; }

protected:
    bool training_ = true;
};

} // namespace nn
