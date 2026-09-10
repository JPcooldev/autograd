#pragma once

#include <cstdint>
#include <vector>
#include <memory>

#include "../node.h"

namespace autograd {

using tensor::Tensor;

/**
 * Copy `t` into a packed row-major buffer by walking its strides and offset.
 * Used so matmul backward sees logical values even when `t` is a view.
 *
 * @param t Tensor to pack.
 * @return Contiguous storage of `t.numel()` elements in row-major order.
 */
template <typename T>
std::vector<T> to_contiguous(const Tensor<T>& t) {
    const int64_t n = t.numel();
    std::vector<T> out(static_cast<size_t>(n));
    if (n == 0)
        return out;

    const int64_t rank     = t.rank();
    const auto&   shape    = t.shape();
    const auto&   strides  = t.strides();
    const int64_t base     = t.offset();
    const auto&   raw      = t.data();
    const auto    contig   = Tensor<T>::compute_contiguous_strides(shape);

    for (int64_t i = 0; i < n; ++i) {
        int64_t remaining = i;
        int64_t pos       = base;
        for (int64_t d = 0; d < rank; ++d) {
            const int64_t idx = remaining / contig[static_cast<size_t>(d)];
            remaining        %= contig[static_cast<size_t>(d)];
            pos              += idx * strides[static_cast<size_t>(d)];
        }
        out[static_cast<size_t>(i)] = raw[static_cast<size_t>(pos)];
    }
    return out;
}

/**
 * Return the batch (leading) dimensions of a matmul operand.
 * Drops the last two axes; empty if rank is below 2.
 *
 * @param shape Operand shape.
 * @return Shape prefix of length `rank - 2`, or `{}`.
 */
inline std::vector<int64_t> matmul_leading_shape(const std::vector<int64_t>& shape) {
    if (shape.size() < 2)
        return {};
    return {shape.begin(), shape.end() - 2};
}

/**
 * Align `src` leading-dim strides to `batch_out`.
 * Missing leading dims and size-1 dims that expand get stride 0 (broadcast).
 *
 * @param src_shape Full shape of the operand (including matrix dims).
 * @param src_strides Strides matching `src_shape`.
 * @param batch_out Broadcast batch shape of the matmul result.
 * @return Strides of length `batch_out.size()`, right-aligned to `src`.
 */
inline std::vector<int64_t> aligned_batch_strides(
    const std::vector<int64_t>& src_shape,
    const std::vector<int64_t>& src_strides,
    const std::vector<int64_t>& batch_out
) {
    const int64_t src_batch_rank = static_cast<int64_t>(src_shape.size()) - 2;
    const int64_t out_batch_rank = static_cast<int64_t>(batch_out.size());
    const int64_t rank_diff      = out_batch_rank - src_batch_rank;

    std::vector<int64_t> out(static_cast<size_t>(out_batch_rank), 0);
    for (int64_t d = 0; d < out_batch_rank; ++d) {
        if (d < rank_diff)
            continue;
        const size_t src = static_cast<size_t>(d - rank_diff);
        if (src_shape[src] == batch_out[static_cast<size_t>(d)])
            out[static_cast<size_t>(d)] = src_strides[src];
    }
    return out;
}

template <typename T>
class DotBackward : public Node<T> {
public:
    /**
     * Construct the backward node for the vector dot product.
     * Aliases `x` and `y` onto `saved_tensors`.
     *
     * @param x First vector.
     * @param y Second vector.
     */
    DotBackward(const Tensor<T>& x, const Tensor<T>& y) : Node<T>(x, y) {}

    /**
     * Compute gradients of z = sum_i x[i] * y[i].
     * Uses saved `x` and `y` and the scalar `propagated_grad.data()[0]`:
     * dL/dx[i] = grad * y[i], dL/dy[i] = grad * x[i].
     *
     * @param propagated_grad Scalar upstream gradient dL/dz.
     * @return `{dL/dx, dL/dy}` with the saved vectors' shapes.
     */
    std::vector<Tensor<T>> apply(const Tensor<T>& propagated_grad) override {
        const T grad_scalar = propagated_grad.data()[0];
        const Tensor<T>& x  = this->saved_tensors[0];
        const Tensor<T>& y  = this->saved_tensors[1];

        const size_t n = static_cast<size_t>(x.numel());
        std::vector<T> grad_x(n);
        std::vector<T> grad_y(n);
        for (size_t i = 0; i < n; ++i) {
            grad_x[i] = grad_scalar * y.data()[i];
            grad_y[i] = grad_scalar * x.data()[i];
        }
        return {
            Tensor<T>::from_operation_result(x.shape(), std::move(grad_x), false, nullptr),
            Tensor<T>::from_operation_result(y.shape(), std::move(grad_y), false, nullptr)
        };
    }
};

template <typename T>
class MatmulBackward : public Node<T> {
    int64_t M_, K_, N_;
    std::vector<T> A_data_;  // contiguous, original A shape
    std::vector<T> B_data_;  // contiguous, original B shape
    std::vector<int64_t> A_shape_;
    std::vector<int64_t> B_shape_;
    std::vector<int64_t> batch_shape_;
    std::vector<int64_t> A_batch_strides_;  // aligned to batch_shape_, contiguous A
    std::vector<int64_t> B_batch_strides_;  // aligned to batch_shape_, contiguous B

public:
    /**
     * Construct the backward node for batched matrix multiply.
     * Aliases A and B for graph edges and stores packed copies plus batch layout
     * so views (e.g. transpose) are handled correctly.
     *
     * @param A Left operand of shape (..., M, K).
     * @param B Right operand of shape (..., K, N).
     * @param batch_shape Broadcast batch dims of the forward result.
     */
    MatmulBackward(const Tensor<T>& A, const Tensor<T>& B, std::vector<int64_t> batch_shape)
        : Node<T>(A, B),
          M_(A.shape()[static_cast<size_t>(A.rank() - 2)]),
          K_(A.shape()[static_cast<size_t>(A.rank() - 1)]),
          N_(B.shape()[static_cast<size_t>(B.rank() - 1)]),
          A_data_(to_contiguous(A)),
          B_data_(to_contiguous(B)),
          A_shape_(A.shape()),
          B_shape_(B.shape()),
          batch_shape_(std::move(batch_shape)),
          A_batch_strides_(aligned_batch_strides(
              A_shape_, Tensor<T>::compute_contiguous_strides(A_shape_), batch_shape_)),
          B_batch_strides_(aligned_batch_strides(
              B_shape_, Tensor<T>::compute_contiguous_strides(B_shape_), batch_shape_)) {}

    /**
     * Compute gradients of C[..., i, j] = sum_k A[..., i, k] * B[..., k, j].
     * Uses packed `A_data_` / `B_data_` and aligned batch strides; sums broadcast
     * batch axes back into A and B. `saved_tensors` values are unused.
     *
     * @param propagated_grad Upstream gradient dL/dC, shape (batch..., M, N).
     * @return `{dL/dA, dL/dB}` with the original A and B shapes.
     */
    std::vector<Tensor<T>> apply(const Tensor<T>& propagated_grad) override {
        const auto& grad = propagated_grad.data(); // contiguous (batch..., M, N)

        const int64_t batch_numel  = Tensor<T>::compute_numel(batch_shape_);
        const auto    batch_contig = Tensor<T>::compute_contiguous_strides(batch_shape_);
        const int64_t batch_rank   = static_cast<int64_t>(batch_shape_.size());

        std::vector<T> dA(static_cast<size_t>(Tensor<T>::compute_numel(A_shape_)), T{0});
        std::vector<T> dB(static_cast<size_t>(Tensor<T>::compute_numel(B_shape_)), T{0});

        for (int64_t b = 0; b < batch_numel; ++b) {
            int64_t remaining = b;
            int64_t A_off     = 0;
            int64_t B_off     = 0;
            for (int64_t d = 0; d < batch_rank; ++d) {
                const int64_t idx = remaining / batch_contig[static_cast<size_t>(d)];
                remaining        %= batch_contig[static_cast<size_t>(d)];
                A_off += idx * A_batch_strides_[static_cast<size_t>(d)];
                B_off += idx * B_batch_strides_[static_cast<size_t>(d)];
            }

            const int64_t C_off = b * M_ * N_;

            // dA[i,k] += Σⱼ grad[i,j] * B[k,j]
            for (int64_t i = 0; i < M_; ++i)
                for (int64_t k = 0; k < K_; ++k)
                    for (int64_t j = 0; j < N_; ++j)
                        dA[static_cast<size_t>(A_off + i * K_ + k)] +=
                            grad[static_cast<size_t>(C_off + i * N_ + j)] *
                            B_data_[static_cast<size_t>(B_off + k * N_ + j)];

            // dB[k,j] += Σᵢ A[i,k] * grad[i,j]
            for (int64_t k = 0; k < K_; ++k)
                for (int64_t j = 0; j < N_; ++j)
                    for (int64_t i = 0; i < M_; ++i)
                        dB[static_cast<size_t>(B_off + k * N_ + j)] +=
                            A_data_[static_cast<size_t>(A_off + i * K_ + k)] *
                            grad[static_cast<size_t>(C_off + i * N_ + j)];
        }
        return {
            Tensor<T>::from_operation_result(A_shape_, std::move(dA), false, nullptr),
            Tensor<T>::from_operation_result(B_shape_, std::move(dB), false, nullptr)
        };
    }
};

} // namespace autograd
