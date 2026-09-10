#pragma once

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "../tensor/tensor.h"
#include "../autograd/grad_context.h"
#include "../autograd/backward_ops/backward_linalg_ops.h"
#include "shape_ops.h"

namespace ops {

/**
 * Compute the dot product of two 1-D tensors. Packs both operands, then sums
 * `x[i] * y[i]` into a scalar tensor of shape `{}` and attaches `DotBackward`
 * when grad is enabled.
 *
 * @param x First 1-D tensor.
 * @param y Second 1-D tensor, same length as `x`.
 * @return A scalar tensor of shape `{}`.
 *
 * @throws std::invalid_argument if either tensor is not 1-D.
 * @throws std::invalid_argument if the tensors have different lengths.
 */
template <typename T>
tensor::Tensor<T> dot(const tensor::Tensor<T>& x, const tensor::Tensor<T>& y) {
    if (x.rank() != 1 || y.rank() != 1)
        throw std::invalid_argument("dot requires two 1D tensors");
    if (x.numel() != y.numel())
        throw std::invalid_argument("dot requires two 1D tensors with the same number of elements");

    T result = T{0};
    const auto xc = contiguous(x);
    const auto yc = contiguous(y);
    const int64_t n = xc.numel();
    for (int64_t i = 0; i < n; ++i)
        result += xc.data()[static_cast<size_t>(i)] * yc.data()[static_cast<size_t>(i)];

    const bool requires_grad = autograd::is_grad_enabled() && (x.requires_grad() || y.requires_grad());
    std::shared_ptr<autograd::Node<T>> grad_fn = nullptr;
    if (requires_grad)
        grad_fn = std::make_shared<autograd::DotBackward<T>>(xc, yc);
    return tensor::Tensor<T>::from_operation_result(
        {}, std::vector<T>{result}, requires_grad, std::move(grad_fn));
}

/**
 * Multiply the last two dimensions of `A` and `B`. Leading dimensions are
 * batch axes and are broadcast together (NumPy / PyTorch rules):
 * `A: (..., M, K)`, `B: (..., K, N)` → `C: (broadcast(...), M, N)`.
 * Element access uses the full stride formula so non-contiguous views are
 * handled correctly, including broadcast batch axes (stride 0).
 *
 * @param A Left operand of rank at least 2.
 * @param B Right operand of rank at least 2.
 * @return The batched matrix product.
 *
 * @throws std::invalid_argument if either argument has rank below 2.
 * @throws std::invalid_argument if the inner dimensions `K` do not match.
 * @throws std::invalid_argument if the batch shapes are not broadcastable.
 */
template <typename T>
tensor::Tensor<T> matmul(const tensor::Tensor<T>& A, const tensor::Tensor<T>& B) {
    if (A.rank() < 2)
        throw std::invalid_argument("matmul: first argument must have rank >= 2");
    if (B.rank() < 2)
        throw std::invalid_argument("matmul: second argument must have rank >= 2");

    // matmul happens only within last two dimensions of the tensors
    //    _________    _______
    //   |         |  |       |
    // M |    A    |  |   B   | K
    //   |_________|  |       |
    //        K       |_______|
    //                    N
    //
    const int64_t M   = A.shape()[static_cast<size_t>(A.rank() - 2)];
    const int64_t K_A = A.shape()[static_cast<size_t>(A.rank() - 1)];
    const int64_t K_B = B.shape()[static_cast<size_t>(B.rank() - 2)];
    const int64_t N   = B.shape()[static_cast<size_t>(B.rank() - 1)];

    if (K_A != K_B)
        throw std::invalid_argument(
            "matmul: inner dimensions must match (" +
            std::to_string(K_A) + " vs " + std::to_string(K_B) + ")");

    const auto batch_out = broadcast_shapes(
        autograd::matmul_leading_shape(A.shape()),
        autograd::matmul_leading_shape(B.shape())
    );

    const auto A_batch_strides = autograd::aligned_batch_strides(
        A.shape(), A.strides(), batch_out);
    const auto B_batch_strides = autograd::aligned_batch_strides(
        B.shape(), B.strides(), batch_out);

    const int64_t A_stride_M     = A.strides()[static_cast<size_t>(A.rank() - 2)];
    const int64_t A_stride_K     = A.strides()[static_cast<size_t>(A.rank() - 1)];
    const int64_t B_stride_K     = B.strides()[static_cast<size_t>(B.rank() - 2)];
    const int64_t B_stride_N     = B.strides()[static_cast<size_t>(B.rank() - 1)];
    const int64_t A_offset = A.offset();
    const int64_t B_offset = B.offset();
    const auto&   A_raw    = A.data();
    const auto&   B_raw    = B.data();

    std::vector<int64_t> out_shape = batch_out;
    out_shape.push_back(M);
    out_shape.push_back(N);

    const int64_t batch_numel  = tensor::Tensor<T>::compute_numel(batch_out);
    const auto    batch_contig = tensor::Tensor<T>::compute_contiguous_strides(batch_out);
    const int64_t batch_rank   = static_cast<int64_t>(batch_out.size());

    std::vector<T> storage(
        static_cast<size_t>(tensor::Tensor<T>::compute_numel(out_shape)), T{0});

    for (int64_t b = 0; b < batch_numel; ++b) {
        int64_t remaining = b;
        int64_t A_base    = A_offset;
        int64_t B_base    = B_offset;
        for (int64_t d = 0; d < batch_rank; ++d) {
            const int64_t idx = remaining / batch_contig[static_cast<size_t>(d)];
            remaining        %= batch_contig[static_cast<size_t>(d)];
            A_base += idx * A_batch_strides[static_cast<size_t>(d)];
            B_base += idx * B_batch_strides[static_cast<size_t>(d)];
        }

        const int64_t C_off = b * M * N;
        for (int64_t i = 0; i < M; ++i)
            for (int64_t k = 0; k < K_A; ++k) {
                const T a_ik = A_raw[static_cast<size_t>(A_base + i * A_stride_M + k * A_stride_K)];
                for (int64_t j = 0; j < N; ++j)
                    storage[static_cast<size_t>(C_off + i * N + j)] +=
                        a_ik * B_raw[static_cast<size_t>(B_base + k * B_stride_K + j * B_stride_N)];
            }
    }

    const bool requires_grad = autograd::is_grad_enabled() && (A.requires_grad() || B.requires_grad());
    std::shared_ptr<autograd::Node<T>> grad_fn = nullptr;
    if (requires_grad)
        grad_fn = std::make_shared<autograd::MatmulBackward<T>>(A, B, batch_out);
    return tensor::Tensor<T>::from_operation_result(
        std::move(out_shape), std::move(storage), requires_grad, std::move(grad_fn));
}

} // namespace ops
