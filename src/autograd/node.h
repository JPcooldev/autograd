#pragma once

#include <vector>
#include <memory>

#include "../tensor/tensor.h"

namespace autograd {

using tensor::Tensor;

template<typename T>
class Node {
public:
    // can operation return multiple outputs that are needed for backpropagation of gradients?
    // virtual std::vector<Tensor<T>> apply(const std::vector<Tensor<T>>& grad_outputs) = 0;
    virtual std::vector<Tensor<T>> apply(const Tensor<T>& propagated_grad) = 0;

    std::vector<std::shared_ptr<Node<T>>> next_edges;   // "children" = input tensors' grad_fn
    std::vector<Tensor<T>> saved_tensors;               // values needed for backward

    Node() = default;

    virtual ~Node() = default;

    // Returns the correct backward edge for tensor x:
    //   - intermediate tensor (has grad_fn)  → its grad_fn
    //   - leaf tensor that requires_grad      → its AccumulateGrad node (created lazily)
    //   - no-grad tensor                      → nullptr (gradient not needed)
    static std::shared_ptr<Node<T>> get_next_edge(const Tensor<T>& x) {
        if (x.grad_fn()) return x.grad_fn();
        if (x.requires_grad()) return x.ensure_accumulate_grad_fn();
        return nullptr;
    }

    explicit Node(const Tensor<T>& x) {
        // Pre-allocate exact capacity so pushes below cannot trigger reallocation.
        saved_tensors.reserve(1);
        next_edges.reserve(1);
        saved_tensors.emplace_back(Tensor<T>::alias(x));
        next_edges.emplace_back(get_next_edge(x));
    }

    Node(const Tensor<T>& x, const Tensor<T>& y) {
        // Pre-allocate exact capacity so pushes below cannot trigger reallocation.
        saved_tensors.reserve(2);
        next_edges.reserve(2);
        saved_tensors.emplace_back(Tensor<T>::alias(x));
        saved_tensors.emplace_back(Tensor<T>::alias(y));
        next_edges.emplace_back(get_next_edge(x));
        next_edges.emplace_back(get_next_edge(y));
    }
};  

} // namespace autograd