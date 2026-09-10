#pragma once

#include <vector>
#include <memory>

#include "../tensor/tensor.h"

namespace autograd {

using tensor::Tensor;

template <typename T>
class Node {
public:
    std::vector<std::shared_ptr<Node<T>>> next_edges;   // "children" = input tensors' grad_fn
    std::vector<Tensor<T>> saved_tensors;               // values needed for backward

    /**
     * Default-construct an empty node with no saved tensors or edges.
     */
    Node() = default;

    /**
     * Destroy the node.
     */
    virtual ~Node() = default;

    /**
     * Compute input gradients from the upstream output gradient.
     * Each subclass implements the local Jacobian of its forward op.
     *
     * @param propagated_grad Upstream gradient with respect to this node's output.
     * @return Per-input gradients, aligned with `next_edges` / `saved_tensors`.
     */
    virtual std::vector<Tensor<T>> apply(const Tensor<T>& propagated_grad) = 0;

    /**
     * Return the backward edge to attach for tensor `x`.
     * Intermediate tensors yield their `grad_fn`; leaves and no-grad tensors
     * yield nullptr so the engine can accumulate via `saved_tensors` or skip.
     *
     * @param x Tensor whose producer edge is requested.
     * @return `x.grad_fn()` if present, otherwise nullptr.
     */
    static std::shared_ptr<Node<T>> get_next_edge(const Tensor<T>& x) {
        if (x.grad_fn())
            return x.grad_fn();
        return nullptr;
    }

    /**
     * Construct a unary node. Aliases `x` into `saved_tensors` and records its
     * next edge, reserving capacity so those pushes cannot reallocate.
     *
     * @param x Forward input to save.
     */
    explicit Node(const Tensor<T>& x) {
        saved_tensors.reserve(1);
        next_edges.reserve(1);
        saved_tensors.emplace_back(Tensor<T>::alias(x));
        next_edges.emplace_back(get_next_edge(x));
    }

    /**
     * Construct a binary node. Aliases `x` then `y` and records both next edges.
     *
     * @param x First forward input to save.
     * @param y Second forward input to save.
     */
    Node(const Tensor<T>& x, const Tensor<T>& y) {
        saved_tensors.reserve(2);
        next_edges.reserve(2);
        saved_tensors.emplace_back(Tensor<T>::alias(x));
        saved_tensors.emplace_back(Tensor<T>::alias(y));
        next_edges.emplace_back(get_next_edge(x));
        next_edges.emplace_back(get_next_edge(y));
    }

    /**
     * Construct a ternary node. Aliases `x`, `y`, then `z` and records all
     * next edges.
     *
     * @param x First forward input to save.
     * @param y Second forward input to save.
     * @param z Third forward input to save.
     */
    Node(const Tensor<T>& x, const Tensor<T>& y, const Tensor<T>& z) {
        saved_tensors.reserve(3);
        next_edges.reserve(3);
        saved_tensors.emplace_back(Tensor<T>::alias(x));
        saved_tensors.emplace_back(Tensor<T>::alias(y));
        saved_tensors.emplace_back(Tensor<T>::alias(z));
        next_edges.emplace_back(get_next_edge(x));
        next_edges.emplace_back(get_next_edge(y));
        next_edges.emplace_back(get_next_edge(z));
    }
};

} // namespace autograd
