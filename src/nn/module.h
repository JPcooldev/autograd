#pragma once

#include <stdexcept>
#include <string>
#include <vector>

#include "layers/layer.h"

namespace nn {

// ---------------------------------------------------------------------------
// Module
// ---------------------------------------------------------------------------
// Base class for composite models (analogous to torch.nn.Module).
//
// Usage
// -----
//   class MLP : public nn::Module<float> {
//   public:
//       nn::Linear<float> fc1{784, 256};
//       nn::Linear<float> fc2{256,  10};
//
//       MLP() {
//           register_module("fc1", fc1);
//           register_module("fc2", fc2);
//       }
//
//       tensor::Tensor<float> forward(const tensor::Tensor<float>& x) const override {
//           return fc2.forward(ops::relu(fc1.forward(x)));
//       }
//   };
//
//   MLP model;
//   SGD<float> opt(model.parameters(), 0.01);
//
// Registration
// ------------
// Because C++ has no __setattr__ hook, sub-modules and bare tensors must be
// registered explicitly in the constructor body via register_module /
// register_parameter. This mirrors the LibTorch C++ frontend convention.
//
// parameters() recurses depth-first through all registered sub-modules and
// collects every tensor that returns requires_grad == true.

template <typename T>
class Module : public Layer<T> {
public:
    // Register a sub-layer / sub-module. Must be called in the constructor.
    // `mod` must outlive this Module (store it as a member).
    void register_module(const std::string& name, Layer<T>& mod) {
        for (auto* existing : submodules_) {
            if (existing == &mod)
                throw std::invalid_argument(
                    "Module::register_module: '" + name + "' is already registered");
        }
        submodules_.push_back(&mod);
    }

    // Register a bare tensor as a learnable parameter directly on this module
    // (e.g., a custom embedding table that isn't wrapped in a Layer).
    void register_parameter(const std::string& name, tensor::Tensor<T>& param) {
        if (!param.requires_grad())
            throw std::invalid_argument(
                "Module::register_parameter: '" + name +
                "' does not have requires_grad=true");
        for (auto* existing : own_params_) {
            if (existing == &param)
                throw std::invalid_argument(
                    "Module::register_parameter: '" + name + "' is already registered");
        }
        own_params_.push_back(&param);
    }

    // Recursively collects all learnable parameters from own_params_ and every
    // registered sub-module, depth-first.
    std::vector<tensor::Tensor<T>*> parameters() override {
        std::vector<tensor::Tensor<T>*> out = own_params_;
        for (auto* mod : submodules_) {
            auto sub = mod->parameters();
            out.insert(out.end(), sub.begin(), sub.end());
        }
        return out;
    }

    // zero_grad for all parameters collected by parameters().
    void zero_grad() {
        for (auto* p : parameters()) {
            if (p->requires_grad())
                p->zero_grad();
        }
    }

private:
    std::vector<Layer<T>*>           submodules_;
    std::vector<tensor::Tensor<T>*>  own_params_;
};

} // namespace nn
