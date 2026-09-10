#pragma once

#include <optional>
#include <string>
#include <utility>
#include <vector>
#include "../../tensor/tensor.h"

namespace nn {

template <typename T>
using NamedTensorList = std::vector<std::pair<std::string, tensor::Tensor<T>*>>;

namespace detail {

/**
 * Build the usual `weight` / optional `bias` named-parameter list.
 *
 * @param weight Learnable weight tensor.
 * @param bias Optional bias; omitted from the list when empty.
 * @return `{{"weight", &weight}, {"bias", &*bias}}` or just weight.
 */
template <typename T>
NamedTensorList<T> named_weight_bias(
    tensor::Tensor<T>& weight,
    std::optional<tensor::Tensor<T>>& bias
) {
    if (bias)
        return {{"weight", &weight}, {"bias", &*bias}};
    return {{"weight", &weight}};
}

/**
 * Strip names from a named-tensor list, preserving order.
 *
 * @param named Named parameter or buffer list.
 * @return Pointers in the same order as `named`.
 */
template <typename T>
std::vector<tensor::Tensor<T>*> pointers_of(const NamedTensorList<T>& named) {
    std::vector<tensor::Tensor<T>*> out;
    out.reserve(named.size());
    for (const auto& kv : named)
        out.push_back(kv.second);
    return out;
}

/**
 * Prefix every name in `named` with `prefix + "."`.
 *
 * @param prefix Sub-module name (must be non-empty).
 * @param named Child named-tensor list.
 * @return New list with dotted names.
 */
template <typename T>
NamedTensorList<T> prefix_names(const std::string& prefix, const NamedTensorList<T>& named) {
    NamedTensorList<T> out;
    out.reserve(named.size());
    for (const auto& kv : named)
        out.emplace_back(prefix + "." + kv.first, kv.second);
    return out;
}

} // namespace detail

// Abstract base class for layers/modules.
//
// Each concrete layer defines its own `forward` signature (Python nn.Module
// does the same). There is no virtual single-tensor forward — losses take
// (input, target) and RNNs return (output, hidden).
//
// Named tensors
// -------------
// `named_parameters()` / `named_buffers()` are the source of truth for
// checkpointing. `parameters()` strips names for the optimizer. `state_dict()`
// concatenates parameters then buffers (PyTorch order). Leaf layers assign
// local names (`weight`, `bias`, `running_mean`, ...). `Module` prefixes child
// names (`fc1.weight`).
template <typename T>
class Layer {
protected:
    bool training_ = true;

public:
    /**
     * Destroy the layer.
     * Virtual so derived layers can be deleted through a Layer pointer.
     */
    virtual ~Layer() = default;

    /**
     * Set training vs evaluation mode.
     * Stores `mode` in `training_`. `Module` overrides this to recurse into sub-layers.
     *
     * @param mode True for training mode, false for evaluation mode.
     */
    virtual void train(bool mode = true) { training_ = mode; }

    /**
     * Switch the layer to evaluation mode.
     * Calls `train(false)`.
     */
    void eval() { train(false); }

    /**
     * Report whether the layer is in training mode.
     *
     * @return True if `training_` is set, false if the layer is in eval mode.
     */
    bool training() const { return training_; }

    /**
     * Expose learnable parameters with local names.
     * The base implementation returns an empty list. Leaf layers return
     * `weight` / `bias` / cell weights; `Module` prefixes registered children.
     *
     * @return `(name, tensor*)` pairs, depth-first.
     */
    virtual NamedTensorList<T> named_parameters() { return {}; }

    /**
     * Expose non-learnable persistent tensors with local names.
     * The base implementation returns an empty list. BatchNorm returns
     * `running_mean` / `running_var`; `Module` prefixes registered children.
     *
     * @return `(name, tensor*)` pairs, depth-first.
     */
    virtual NamedTensorList<T> named_buffers() { return {}; }

    /**
     * Expose learnable parameter tensors.
     * Strips names from `named_parameters()`; used by optimizers.
     *
     * @return Pointers to learnable parameters (empty by default).
     */
    std::vector<tensor::Tensor<T>*> parameters() {
        return detail::pointers_of(named_parameters());
    }

    /**
     * Collect parameters then buffers under dotted names.
     * Concatenates `named_parameters()` and `named_buffers()`.
     *
     * @return Full named state used by `nn::save` / `nn::load`.
     */
    NamedTensorList<T> state_dict() {
        NamedTensorList<T> out = named_parameters();
        auto bufs = named_buffers();
        out.insert(out.end(), bufs.begin(), bufs.end());
        return out;
    }
};

} // namespace nn
