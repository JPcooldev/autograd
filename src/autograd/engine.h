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

// DFS post-order traversal starting from `root`.
// Each node is visited at most once (cycles are impossible in a DAG).
// Nodes are pushed onto `order` in post-order, so the front of `order`
// when reversed is the topological order (root first, leaves last).
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

// Element-wise in-place addition: dst += src.
// Both tensors must have the same number of elements.
template <typename T>
void accumulate_into(tensor::Tensor<T>& dst, const tensor::Tensor<T>& src) {
    auto& d = dst.data();
    const auto& s = src.data();
    for (size_t i = 0; i < d.size(); ++i)
        d[i] += s[i];
}

} // namespace detail

// Execute the backward pass starting from root_fn with the given seed gradient.
//
// Algorithm:
//   1. Topological sort (DFS post-order, then reverse) so we process nodes
//      from output toward inputs.
//   2. Maintain an accumulator map: Node* → accumulated gradient Tensor.
//      Seed it with root_fn → initial_grad.
//   3. For each node in order: call apply(accumulated_grad), then distribute
//      the returned per-input gradients to next_edges accumulators.
//   4. AccumulateGrad::apply() stores the final gradient on the leaf tensor.
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
            if (!next)
                continue;

            auto acc_it = accumulators.find(next);
            if (acc_it == accumulators.end()) {
                accumulators.emplace(next,
                    tensor::Tensor<T>(input_grads[i].shape(),
                                      input_grads[i].data(), false));
            } else {
                detail::accumulate_into(acc_it->second, input_grads[i]);
            }
        }
    }
}

} // namespace autograd
