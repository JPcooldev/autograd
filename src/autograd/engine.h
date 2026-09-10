#pragma once

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "node.h"

namespace autograd {

namespace detail {

/**
 * Depth-first post-order walk of the backward DAG from `node`.
 * Each pointer is inserted into `visited` at most once; nodes are appended to
 * `order` after their `next_edges` so reversing `order` is output-first topo.
 *
 * @param node Current node, or nullptr (ignored).
 * @param visited Set of already-walked node pointers.
 * @param order Accumulator filled in post-order (leaves first).
 */
template <typename T>
void topo_dfs(
    Node<T>* node,
    std::unordered_set<Node<T>*>& visited,
    std::vector<Node<T>*>& order
) {
    if (!node || visited.count(node))
        return;
    visited.insert(node);
    for (const auto& edge : node->next_edges)
        topo_dfs(edge.get(), visited, order);
    // TODO: check if we cannot use emplace_back so we don't use copy constructor
    order.push_back(node);
}

/**
 * Add `src` into `dst` element-wise in place (`dst += src`).
 * Packs `src` to a contiguous buffer, then adds into `dst.data()`.
 *
 * @param dst Destination tensor; mutated.
 * @param src Source tensor; must have the same number of elements as `dst`.
 */
template <typename T>
void accumulate_into(tensor::Tensor<T>& dst, const tensor::Tensor<T>& src) {
    const auto packed = src.contiguous();
    auto& d = dst.data();
    const auto& s = packed.data();
    for (size_t i = 0; i < d.size(); ++i)
        d[i] += s[i];
}

} // namespace detail

/**
 * Run reverse-mode autodiff from `root_fn` with seed gradient `initial_grad`.
 * Topo-sorts the graph, accumulates per-node output grads, calls `apply`,
 * and either forwards to the next node or `accumulate_grad`s on requiring-grad
 * leaves (nullptr edges).
 *
 * @param root_fn Backward node of the output tensor.
 * @param initial_grad Seed gradient for that output (typically ones for a scalar).
 *
 * @throws std::invalid_argument if `root_fn` is null.
 * @throws std::invalid_argument if `initial_grad`'s storage size does not match
 *         its shape (when copying the seed into the accumulator map).
 */
template <typename T>
void run_backward(
    const std::shared_ptr<Node<T>>& root_fn,
    const tensor::Tensor<T>& initial_grad
) {
    if (!root_fn)
        throw std::invalid_argument("run_backward: root_fn is null");

    // Step 1 – topological sort (post-order DFS → reverse = output-first order)
    std::unordered_set<Node<T>*> visited;
    std::vector<Node<T>*> order;       // post-order (leaves first)
    detail::topo_dfs(root_fn.get(), visited, order);
    // Reverse so we process the output node first.
    std::reverse(order.begin(), order.end());

    // Step 2 – seed the accumulator map
    std::unordered_map<Node<T>*, tensor::Tensor<T>> accumulators;
    accumulators.emplace(root_fn.get(),
        tensor::Tensor<T>(initial_grad.shape(), initial_grad.data(), false));

    // Step 3 – backward loop
    for (Node<T>* node : order) {
        auto it = accumulators.find(node);
        if (it == accumulators.end())
            continue;  // no gradient reached this node (can happen in mixed-grad graphs)

        const tensor::Tensor<T>& node_grad = it->second;
        const std::vector<tensor::Tensor<T>> input_grads = node->apply(node_grad);

        const size_t n_edges = node->next_edges.size();
        for (size_t i = 0; i < n_edges && i < input_grads.size(); ++i) {
            Node<T>* next = node->next_edges[i].get();
            if (next) {
                // Non-leaf: propagate gradient to the next backward node.
                auto acc_it = accumulators.find(next);
                if (acc_it == accumulators.end()) {
                    const auto packed = input_grads[i].contiguous();
                    accumulators.emplace(next,
                        tensor::Tensor<T>(packed.shape(), packed.data(), false));
                } else {
                    detail::accumulate_into(acc_it->second, input_grads[i]);
                }
            } else if (i < node->saved_tensors.size() &&
                       node->saved_tensors[i].requires_grad()) {
                // Leaf input: accumulate gradient directly into the shared
                // grad buffer (visible on the original tensor via tensor.grad()).
                node->saved_tensors[i].accumulate_grad(input_grads[i]);
            }
        }
    }
}

} // namespace autograd
