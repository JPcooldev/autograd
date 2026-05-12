#pragma once

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

#include "../tensor/tensor.h"
#include "../autograd/grad_context.h"
#include "../autograd/backward_ops/backward_linalg_ops.h"

namespace ops {

// ----- dot product -----
// Both x and y must be 1D tensors of the same size.
// Returns a scalar tensor of shape {}.
template <typename T>
tensor::Tensor<T> dot(const tensor::Tensor<T>& x, const tensor::Tensor<T>& y) {
    if (x.rank() != 1 || y.rank() != 1)
        throw std::invalid_argument("dot requires two 1D tensors");
    if (x.numel() != y.numel())
        throw std::invalid_argument(
            "dot requires tensors with the same number of elements");

    T result = T{0};
    const int64_t n = x.numel();
    for (int64_t i = 0; i < n; ++i)
        result += x.data()[static_cast<size_t>(i)] * y.data()[static_cast<size_t>(i)];

    const bool requires_grad = autograd::is_grad_enabled() &&
                               (x.requires_grad() || y.requires_grad());
    std::shared_ptr<autograd::Node<T>> grad_fn = nullptr;
    if (requires_grad)
        grad_fn = std::make_shared<autograd::DotBackward<T>>(x, y);

    return tensor::Tensor<T>::from_operation_result(
        {}, std::vector<T>{result}, requires_grad, std::move(grad_fn));
}

// ----- matrix multiplication -----
// A must be 2D (M, K); B must be 2D (K, N). Returns a 2D tensor (M, N).
//
// Element access uses the full stride formula A[i,k] = data[offset + i*s0 + k*s1]
// so non-contiguous views (e.g. after transpose()) are handled correctly.
template <typename T>
tensor::Tensor<T> matmul(const tensor::Tensor<T>& A, const tensor::Tensor<T>& B) {
    if (A.rank() != 2)
        throw std::invalid_argument("matmul: first argument must be a 2D tensor");
    if (B.rank() != 2)
        throw std::invalid_argument("matmul: second argument must be a 2D tensor");

    const int64_t M = A.shape()[0];
    const int64_t K = A.shape()[1];
    const int64_t K2 = B.shape()[0];
    const int64_t N  = B.shape()[1];

    if (K != K2)
        throw std::invalid_argument(
            "matmul: inner dimensions must match (" +
            std::to_string(K) + " vs " + std::to_string(K2) + ")");

    const int64_t A_offset = A.offset();
    const int64_t A_s0     = A.strides()[0];
    const int64_t A_s1     = A.strides()[1];
    const int64_t B_offset = B.offset();
    const int64_t B_s0     = B.strides()[0];
    const int64_t B_s1     = B.strides()[1];
    const auto&   A_raw    = A.data();
    const auto&   B_raw    = B.data();

    std::vector<T> storage(static_cast<size_t>(M * N), T{0});
    for (int64_t i = 0; i < M; ++i)
        for (int64_t k = 0; k < K; ++k) {
            const T a_ik = A_raw[static_cast<size_t>(A_offset + i * A_s0 + k * A_s1)];
            for (int64_t j = 0; j < N; ++j)
                storage[static_cast<size_t>(i * N + j)] +=
                    a_ik * B_raw[static_cast<size_t>(B_offset + k * B_s0 + j * B_s1)];
        }

    const bool requires_grad = autograd::is_grad_enabled() &&
                               (A.requires_grad() || B.requires_grad());
    std::shared_ptr<autograd::Node<T>> grad_fn = nullptr;
    if (requires_grad)
        grad_fn = std::make_shared<autograd::MatmulBackward<T>>(A, B);

    return tensor::Tensor<T>::from_operation_result(
        {M, N}, std::move(storage), requires_grad, std::move(grad_fn));
}

} // namespace ops