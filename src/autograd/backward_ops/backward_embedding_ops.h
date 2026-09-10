#pragma once

#include <cstdint>
#include <vector>

#include "../node.h"

namespace autograd {

using tensor::Tensor;

template <typename T>
class EmbeddingBackward : public Node<T> {
    std::vector<int64_t> indices_;
    int64_t embedding_dim_;
    std::vector<int64_t> weight_shape_;

public:
    /**
     * Construct the backward node for embedding lookup.
     * Aliases `weight` for the graph edge and stores packed integer indices
     * plus the embedding width so `apply` can scatter-add into `dW`.
     *
     * @param weight Forward embedding table of shape `{V, D}`.
     * @param indices Packed lookup indices of length `n` (row-major over the
     *        index tensor).
     * @param embedding_dim Table width `D`.
     */
    EmbeddingBackward(
        const Tensor<T>& weight,
        std::vector<int64_t> indices,
        int64_t embedding_dim
    ) :
        Node<T>(weight),
        indices_(std::move(indices)),
        embedding_dim_(embedding_dim),
        weight_shape_(weight.shape()) {}

    /**
     * Scatter-add output gradients into a dense table gradient.
     * For each lookup `i`, adds `grad[i, :]` into row `indices_[i]` of a
     * zero `{V, D}` tensor. Repeated indices accumulate.
     *
     * @param propagated_grad Upstream gradient dL/d(embedding output).
     * @return `{dL/d(weight)}`.
     */
    std::vector<Tensor<T>> apply(const Tensor<T>& propagated_grad) override {
        Tensor<T> grad_weight = Tensor<T>::zeros(weight_shape_, false);
        const Tensor<T> g = propagated_grad.contiguous();
        auto& dst = grad_weight.data();
        const auto& src = g.data();
        const int64_t n = static_cast<int64_t>(indices_.size());
        for (int64_t i = 0; i < n; ++i) {
            const int64_t row = indices_[static_cast<size_t>(i)];
            const int64_t src_off = i * embedding_dim_;
            const int64_t dst_off = row * embedding_dim_;
            for (int64_t d = 0; d < embedding_dim_; ++d)
                dst[static_cast<size_t>(dst_off + d)] +=
                    src[static_cast<size_t>(src_off + d)];
        }
        return {grad_weight};
    }
};

} // namespace autograd
