#pragma once

#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "../../tensor/tensor.h"
#include "layer.h"
#include "../../ops/embedding_ops.h"

namespace nn {

// ---------------------------------------------------------------------------
// Embedding
// ---------------------------------------------------------------------------
// Lookup table:  y = weight[indices]
//
// Parameters
//   weight  shape {num_embeddings, embedding_dim}  — N(0, 1)  (PyTorch default)
//
// Forward input: integer tensor of any rank (`int32_t` or `int64_t`).
// Output shape:  indices.shape + {embedding_dim}.

template <typename T>
class Embedding : public Layer<T> {
private:
    int64_t num_embeddings_;
    int64_t embedding_dim_;

public:
    // {num_embeddings, embedding_dim}
    tensor::Tensor<T> weight;

    /**
     * Construct a learnable embedding table.
     * Initializes `weight` with i.i.d. N(0, 1) samples (`requires_grad=true`).
     *
     * @param num_embeddings Number of rows V (vocabulary size).
     * @param embedding_dim Width D of each embedding vector.
     *
     * @throws std::invalid_argument if `num_embeddings` is not > 0.
     * @throws std::invalid_argument if `embedding_dim` is not > 0.
     */
    Embedding(int64_t num_embeddings, int64_t embedding_dim)
        : num_embeddings_(num_embeddings),
          embedding_dim_(embedding_dim),
          weight(tensor::Tensor<T>::randn({num_embeddings, embedding_dim}, true)) {
        if (num_embeddings <= 0)
            throw std::invalid_argument("Embedding: num_embeddings must be > 0");
        if (embedding_dim <= 0)
            throw std::invalid_argument("Embedding: embedding_dim must be > 0");
    }

    /**
     * Gather rows of `weight` at `indices`.
     * Delegates to `ops::embedding`. Integer indices are not differentiated.
     *
     * @param indices Integer tensor (`int32_t` or `int64_t`) of any rank.
     * @return Embeddings with shape `indices.shape + {embedding_dim}`.
     *
     * @throws std::invalid_argument if any index is outside `[0, num_embeddings)`.
     */
    template <typename Index>
    typename std::enable_if<
        std::is_same<Index, int32_t>::value || std::is_same<Index, int64_t>::value,
        tensor::Tensor<T>>::type
    forward(const tensor::Tensor<Index>& indices) const {
        return ops::embedding(weight, indices);
    }

    /**
     * Return the number of embedding rows.
     *
     * @return `num_embeddings_`.
     */
    int64_t num_embeddings() const { return num_embeddings_; }

    /**
     * Return the embedding vector width.
     *
     * @return `embedding_dim_`.
     */
    int64_t embedding_dim() const { return embedding_dim_; }

    /**
     * Return a named pointer to the embedding table.
     *
     * @return `{{"weight", &weight}}`.
     */
    NamedTensorList<T> named_parameters() override { return {{"weight", &weight}}; }
};

} // namespace nn
