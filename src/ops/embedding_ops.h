#pragma once

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "../tensor/tensor.h"
#include "../autograd/grad_context.h"
#include "../autograd/backward_ops/backward_embedding_ops.h"
#include "shape_ops.h"

namespace ops {

/**
 * Gather rows of an embedding table.
 * `weight` is `{V, D}`; each integer in `indices` selects a row. The output
 * shape is `indices.shape + {D}`. Attaches `EmbeddingBackward` when grad is
 * enabled and `weight` requires grad. Indices are not differentiated.
 *
 * @param weight Rank-2 table `{num_embeddings, embedding_dim}`.
 * @param indices Integer tensor (`int32_t` or `int64_t`) of any rank.
 * @return Gathered embeddings with one trailing feature axis of size `D`.
 *
 * @throws std::invalid_argument if `weight` is not rank 2.
 * @throws std::invalid_argument if `Index` is not `int32_t` or `int64_t`.
 * @throws std::invalid_argument if any index is outside `[0, V)`.
 */
template <typename T, typename Index>
typename std::enable_if<
    std::is_same<Index, int32_t>::value || std::is_same<Index, int64_t>::value,
    tensor::Tensor<T>>::type
embedding(
    const tensor::Tensor<T>& weight,
    const tensor::Tensor<Index>& indices
) {
    if (weight.rank() != 2)
        throw std::invalid_argument(
            "embedding: weight must have shape {num_embeddings, embedding_dim}");

    const int64_t vocab = weight.shape()[0];
    const int64_t dim = weight.shape()[1];
    const auto w = contiguous(weight);
    const auto idx = contiguous(indices);
    const int64_t n = idx.numel();

    std::vector<int64_t> ids(static_cast<size_t>(n));
    const auto& idata = idx.data();
    for (int64_t i = 0; i < n; ++i) {
        const int64_t row = static_cast<int64_t>(idata[static_cast<size_t>(i)]);
        if (row < 0 || row >= vocab)
            throw std::invalid_argument(
                "embedding: index " + std::to_string(row) +
                " is out of range for vocab size " + std::to_string(vocab));
        ids[static_cast<size_t>(i)] = row;
    }

    std::vector<T> storage(static_cast<size_t>(n * dim));
    const auto& wd = w.data();
    for (int64_t i = 0; i < n; ++i) {
        const int64_t src_off = ids[static_cast<size_t>(i)] * dim;
        const int64_t dst_off = i * dim;
        for (int64_t d = 0; d < dim; ++d)
            storage[static_cast<size_t>(dst_off + d)] =
                wd[static_cast<size_t>(src_off + d)];
    }

    std::vector<int64_t> out_shape = indices.shape();
    out_shape.push_back(dim);

    const bool requires_grad = autograd::is_grad_enabled() && weight.requires_grad();
    std::shared_ptr<autograd::Node<T>> grad_fn = nullptr;
    if (requires_grad)
        grad_fn = std::make_shared<autograd::EmbeddingBackward<T>>(weight, std::move(ids), dim);

    return tensor::Tensor<T>::from_operation_result(
        out_shape, std::move(storage), requires_grad, std::move(grad_fn));
}

} // namespace ops
