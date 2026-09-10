#pragma once

#include <stdexcept>
#include <string>
#include <utility>
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
//       tensor::Tensor<float> forward(const tensor::Tensor<float>& x) const {
//           return fc2.forward(ops::relu(fc1.forward(x)));
//       }
//   };
//
//   MLP model;
//   SGD<float> opt(model.parameters(), 0.01);
//   nn::save(model, "mlp.agck");
//
// Registration
// ------------
// Because C++ has no __setattr__ hook, sub-modules and bare tensors must be
// registered explicitly in the constructor body via register_module /
// register_parameter / register_buffer. This mirrors the LibTorch C++ frontend
// convention. Names are stored and become dotted `state_dict` keys
// (`fc1.weight`). Duplicate names at the same module level throw.
//
// named_parameters() recurses depth-first: own parameters, then each
// registered sub-module with names prefixed. parameters() strips names.
// named_buffers() is the same walk for non-learnable persistent tensors.
// num_parameters() sums numel() over parameters() only.

template <typename T>
class Module : public Layer<T> {
private:
    std::vector<std::pair<std::string, Layer<T>*>>           submodules_;
    std::vector<std::pair<std::string, tensor::Tensor<T>*>>  own_params_;
    std::vector<std::pair<std::string, tensor::Tensor<T>*>>  own_buffers_;

    /**
     * Reject an empty registration name.
     *
     * @param kind "module", "parameter", or "buffer" for the error message.
     * @param name Candidate name.
     *
     * @throws std::invalid_argument if `name` is empty.
     */
    static void require_name(const char* kind, const std::string& name) {
        if (name.empty())
            throw std::invalid_argument(
                std::string("Module::register_") + kind + ": name must be non-empty");
    }

    /**
     * Reject a name that is already used by a sibling registration.
     *
     * @param kind "module", "parameter", or "buffer" for the error message.
     * @param name Candidate name.
     *
     * @throws std::invalid_argument if `name` is already registered on this module.
     */
    void require_unique_name(const char* kind, const std::string& name) const {
        for (const auto& kv : submodules_)
            if (kv.first == name)
                throw std::invalid_argument(
                    std::string("Module::register_") + kind + ": '" + name +
                    "' is already registered");
        for (const auto& kv : own_params_)
            if (kv.first == name)
                throw std::invalid_argument(
                    std::string("Module::register_") + kind + ": '" + name +
                    "' is already registered");
        for (const auto& kv : own_buffers_)
            if (kv.first == name)
                throw std::invalid_argument(
                    std::string("Module::register_") + kind + ": '" + name +
                    "' is already registered");
    }

public:
    /**
     * Register a sub-layer / sub-module.
     * Appends `(name, &mod)` to `submodules_` so `train`, `named_parameters`,
     * and `named_buffers` recurse into it. `mod` must outlive this Module
     * (store it as a member).
     *
     * @param name Local name; becomes the prefix of child state_dict keys.
     * @param mod The sub-layer to register; must remain valid for this Module's lifetime.
     *
     * @throws std::invalid_argument if `name` is empty.
     * @throws std::invalid_argument if `name` is already registered.
     * @throws std::invalid_argument if `mod` is already registered.
     */
    void register_module(const std::string& name, Layer<T>& mod) {
        require_name("module", name);
        require_unique_name("module", name);
        for (const auto& existing : submodules_)
            if (existing.second == &mod)
                throw std::invalid_argument(
                    "Module::register_module: '" + name + "' is already registered");
        submodules_.emplace_back(name, &mod);
    }

    /**
     * Register a bare tensor as a learnable parameter on this module.
     * Appends `(name, &param)` to `own_params_` (e.g. a custom embedding table
     * not wrapped in a Layer).
     *
     * @param name Local state_dict key.
     * @param param Tensor to register; must have `requires_grad == true` and outlive this Module.
     *
     * @throws std::invalid_argument if `name` is empty.
     * @throws std::invalid_argument if `name` is already registered.
     * @throws std::invalid_argument if `param` does not have `requires_grad == true`.
     * @throws std::invalid_argument if `param` is already registered.
     */
    void register_parameter(const std::string& name, tensor::Tensor<T>& param) {
        require_name("parameter", name);
        require_unique_name("parameter", name);
        if (!param.requires_grad())
            throw std::invalid_argument(
                "Module::register_parameter: '" + name +
                "' does not have requires_grad=true");
        for (const auto& existing : own_params_)
            if (existing.second == &param)
                throw std::invalid_argument(
                    "Module::register_parameter: '" + name + "' is already registered");
        own_params_.emplace_back(name, &param);
    }

    /**
     * Register a non-learnable persistent tensor (BatchNorm-style running stats).
     * Appends `(name, &buf)` to `own_buffers_`. Buffers appear in `state_dict()`
     * after parameters and are copied by `nn::save` / `nn::load`.
     *
     * @param name Local state_dict key.
     * @param buf Tensor to register; must outlive this Module.
     *
     * @throws std::invalid_argument if `name` is empty.
     * @throws std::invalid_argument if `name` is already registered.
     * @throws std::invalid_argument if `buf` is already registered.
     */
    void register_buffer(const std::string& name, tensor::Tensor<T>& buf) {
        require_name("buffer", name);
        require_unique_name("buffer", name);
        for (const auto& existing : own_buffers_)
            if (existing.second == &buf)
                throw std::invalid_argument(
                    "Module::register_buffer: '" + name + "' is already registered");
        own_buffers_.emplace_back(name, &buf);
    }

    /**
     * Recurse train/eval into every registered sub-module.
     * Sets this module's `training_` flag, then calls `train(mode)` on each entry in `submodules_`.
     *
     * @param mode True for training mode, false for evaluation mode.
     */
    void train(bool mode = true) override {
        Layer<T>::train(mode);
        for (auto& kv : submodules_)
            kv.second->train(mode);
    }

    /**
     * Collect named learnable parameters from this module and its sub-modules.
     * Starts from `own_params_`, then depth-first concatenates each child's
     * `named_parameters()` with the child's registration name as prefix.
     *
     * @return Dotted `(name, tensor*)` pairs.
     */
    NamedTensorList<T> named_parameters() override {
        NamedTensorList<T> out = own_params_;
        for (auto& kv : submodules_) {
            auto sub = detail::prefix_names(kv.first, kv.second->named_parameters());
            out.insert(out.end(), sub.begin(), sub.end());
        }
        return out;
    }

    /**
     * Collect named buffers from this module and its sub-modules.
     * Starts from `own_buffers_`, then depth-first concatenates each child's
     * `named_buffers()` with the child's registration name as prefix.
     *
     * @return Dotted `(name, tensor*)` pairs.
     */
    NamedTensorList<T> named_buffers() override {
        NamedTensorList<T> out = own_buffers_;
        for (auto& kv : submodules_) {
            auto sub = detail::prefix_names(kv.first, kv.second->named_buffers());
            out.insert(out.end(), sub.begin(), sub.end());
        }
        return out;
    }

    /**
     * Count scalar values across all learnable parameters.
     * Iterates `parameters()` and sums each tensor's `numel()`.
     *
     * @return Total number of elements over every tensor returned by `parameters()`.
     */
    int64_t num_parameters() {
        int64_t n = 0;
        for (auto* p : this->parameters())
            n += p->numel();
        return n;
    }

    /**
     * Zero gradients of all collected parameters.
     * Iterates `parameters()` and calls `zero_grad()` on each tensor with `requires_grad`.
     */
    void zero_grad() {
        for (auto* p : this->parameters())
            if (p->requires_grad())
                p->zero_grad();
    }
};

} // namespace nn
