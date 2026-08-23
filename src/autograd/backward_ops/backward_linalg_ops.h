#pragma once

#include <cstdint>
#include <vector>
#include <memory>

#include "../node.h"

namespace autograd {

using tensor::Tensor;

// Build a contiguous row-major copy of a 2D tensor by reading through its
// actual strides and offset. This is needed for matmul inputs that may be
// non-contiguous views (e.g. the result of transpose()).
template <typename T>
std::vector<T> to_contiguous_2d(const Tensor<T>& t) {
    const int64_t rows       = t.shape()[0];
    const int64_t cols       = t.shape()[1];
    const int64_t row_stride = t.strides()[0];
    const int64_t col_stride = t.strides()[1];
    const int64_t base       = t.offset();
    const auto&   raw        = t.data();

    std::vector<T> out(static_cast<size_t>(rows * cols));
    for (int64_t i = 0; i < rows; ++i)
        for (int64_t j = 0; j < cols; ++j)
            out[static_cast<size_t>(i * cols + j)] =
                raw[static_cast<size_t>(base + i * row_stride + j * col_stride)];
    return out;
}

// ----- dot backward -----
// Forward: scalar = Σ xᵢ * yᵢ
// Backward:
//   dL/dx[i] = grad_scalar * y[i]
//   dL/dy[i] = grad_scalar * x[i]
template <typename T>
class DotBackward : public Node<T> {
public:
    DotBackward(const Tensor<T>& x, const Tensor<T>& y) : Node<T>(x, y) {}

    std::vector<Tensor<T>> apply(const Tensor<T>& propagated_grad) override {
        const T grad_scalar = propagated_grad.data()[0];
        const Tensor<T>& x  = this->saved_tensors[0];
        const Tensor<T>& y  = this->saved_tensors[1];

        const int64_t n = x.numel();
        std::vector<T> grad_x(static_cast<size_t>(n));
        std::vector<T> grad_y(static_cast<size_t>(n));
        for (int64_t i = 0; i < n; ++i) {
            grad_x[static_cast<size_t>(i)] = grad_scalar * y.data()[static_cast<size_t>(i)];
            grad_y[static_cast<size_t>(i)] = grad_scalar * x.data()[static_cast<size_t>(i)];
        }
        return {
            Tensor<T>::from_operation_result(x.shape(), std::move(grad_x), false, nullptr),
            Tensor<T>::from_operation_result(y.shape(), std::move(grad_y), false, nullptr)
        };
    }
};

// ----- matmul backward -----
// Forward: C = A @ B,  A: (M, K),  B: (K, N),  C: (M, N)
// Backward:
//   dL/dA[i,k] = Σⱼ dL/dC[i,j] * B[k,j]   → shape (M, K)
//   dL/dB[k,j] = Σᵢ A[i,k]     * dL/dC[i,j] → shape (K, N)
//
// A and B are stored as contiguous copies (to_contiguous_2d) so that
// non-contiguous inputs (e.g. after transpose()) are handled correctly.
template <typename T>
class MatmulBackward : public Node<T> {
    int64_t M_, K_, N_;
    std::vector<T> A_data_;  // contiguous (M x K)
    std::vector<T> B_data_;  // contiguous (K x N)
    std::vector<int64_t> A_shape_;
    std::vector<int64_t> B_shape_;

public:
    MatmulBackward(const Tensor<T>& A, const Tensor<T>& B)
        : Node<T>(A, B),
          M_(A.shape()[0]), K_(A.shape()[1]), N_(B.shape()[1]),
          A_data_(to_contiguous_2d(A)),
          B_data_(to_contiguous_2d(B)),
          A_shape_(A.shape()),
          B_shape_(B.shape())
    {
        // Node<T>(A, B) sets saved_tensors[0/1] and next_edges[0/1].
        // A_data_ / B_data_ are contiguous copies needed for the actual
        // backward computation (inputs may be non-contiguous after transpose).
    }

    std::vector<Tensor<T>> apply(const Tensor<T>& propagated_grad) override {
        const auto& grad = propagated_grad.data(); // contiguous (M x N)

        // dL/dA (M x K): dA[i,k] = Σⱼ grad[i,j] * B[k,j]
        std::vector<T> dA(static_cast<size_t>(M_ * K_), T{0});
        for (int64_t i = 0; i < M_; ++i)
            for (int64_t k = 0; k < K_; ++k)
                for (int64_t j = 0; j < N_; ++j)
                    dA[static_cast<size_t>(i * K_ + k)] +=
                        grad[static_cast<size_t>(i * N_ + j)] *
                        B_data_[static_cast<size_t>(k * N_ + j)];

        // dL/dB (K x N): dB[k,j] = Σᵢ A[i,k] * grad[i,j]
        std::vector<T> dB(static_cast<size_t>(K_ * N_), T{0});
        for (int64_t k = 0; k < K_; ++k)
            for (int64_t j = 0; j < N_; ++j)
                for (int64_t i = 0; i < M_; ++i)
                    dB[static_cast<size_t>(k * N_ + j)] +=
                        A_data_[static_cast<size_t>(i * K_ + k)] *
                        grad[static_cast<size_t>(i * N_ + j)];

        return {
            Tensor<T>::from_operation_result(A_shape_, std::move(dA), false, nullptr),
            Tensor<T>::from_operation_result(B_shape_, std::move(dB), false, nullptr)
        };
    }
};

} // namespace autograd
