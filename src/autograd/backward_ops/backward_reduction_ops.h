#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "../node.h"

namespace autograd {

using tensor::Tensor;

// ----- sum() backward -----
// Forward: scalar = sum of all elements.
// Backward: every input element receives propagated_grad unchanged.
template <typename T>
class SumBackward : public Node<T> {
    std::vector<int64_t> input_shape_;
public:
    explicit SumBackward(const Tensor<T>& x)
        : Node<T>(), input_shape_(x.shape())
    {
        this->next_edges.reserve(1);
        this->next_edges.emplace_back(Node<T>::get_next_edge(x));
    }

    std::vector<Tensor<T>> apply(const Tensor<T>& propagated_grad) override
    {
        const T grad_val = propagated_grad.data()[0];
        const int64_t numel = Tensor<T>::compute_numel(input_shape_);
        std::vector<T> out(static_cast<size_t>(numel), grad_val);
        return {Tensor<T>::from_operation_result(input_shape_, std::move(out), false, nullptr)};
    }
};

// ----- sum(dim) backward -----
// Forward: output shape = input shape with dim removed.
// Backward: each input element receives the propagated_grad of the output
//           element it contributed to (broadcast the grad back along dim).
template <typename T>
class SumDimBackward : public Node<T> {
    std::vector<int64_t> input_shape_;
    int64_t dim_;
public:
    SumDimBackward(const Tensor<T>& x, int64_t dim)
        : Node<T>(), input_shape_(x.shape()), dim_(dim)
    {
        this->next_edges.reserve(1);
        this->next_edges.emplace_back(Node<T>::get_next_edge(x));
    }

    std::vector<Tensor<T>> apply(const Tensor<T>& propagated_grad) override
    {
        const int64_t in_rank  = static_cast<int64_t>(input_shape_.size());
        const int64_t in_numel = Tensor<T>::compute_numel(input_shape_);
        const auto in_strides  = Tensor<T>::compute_contiguous_strides(input_shape_);

        // out_shape = input_shape with dim_ removed
        std::vector<int64_t> out_shape;
        out_shape.reserve(static_cast<size_t>(in_rank - 1));
        for (int64_t d = 0; d < in_rank; ++d)
            if (d != dim_) out_shape.push_back(input_shape_[static_cast<size_t>(d)]);

        const auto out_strides = Tensor<T>::compute_contiguous_strides(out_shape);

        std::vector<T> grad_in(static_cast<size_t>(in_numel));
        for (int64_t flat = 0; flat < in_numel; ++flat) {
            int64_t remaining = flat;
            int64_t out_flat  = 0;
            int64_t out_d     = 0;
            for (int64_t d = 0; d < in_rank; ++d) {
                const int64_t idx = remaining / in_strides[static_cast<size_t>(d)];
                remaining        %= in_strides[static_cast<size_t>(d)];
                if (d != dim_) {
                    out_flat += idx * out_strides[static_cast<size_t>(out_d)];
                    ++out_d;
                }
            }
            grad_in[static_cast<size_t>(flat)] = propagated_grad.data()[static_cast<size_t>(out_flat)];
        }
        return {Tensor<T>::from_operation_result(input_shape_, std::move(grad_in), false, nullptr)};
    }
};

// ----- mean() backward -----
// Forward: scalar = sum / numel.
// Backward: every input element receives propagated_grad / numel.
template <typename T>
class MeanBackward : public Node<T> {
    std::vector<int64_t> input_shape_;
    int64_t numel_;
public:
    explicit MeanBackward(const Tensor<T>& x)
        : Node<T>(), input_shape_(x.shape()), numel_(x.numel())
    {
        this->next_edges.reserve(1);
        this->next_edges.emplace_back(Node<T>::get_next_edge(x));
    }

    std::vector<Tensor<T>> apply(const Tensor<T>& propagated_grad) override
    {
        const T grad_val = propagated_grad.data()[0] / static_cast<T>(numel_);
        std::vector<T> out(static_cast<size_t>(numel_), grad_val);
        return {Tensor<T>::from_operation_result(input_shape_, std::move(out), false, nullptr)};
    }
};

// ----- mean(dim) backward -----
// Forward: output shape = input shape with dim removed; values divided by dim_size.
// Backward: same expansion as SumDimBackward, then divided by dim_size.
template <typename T>
class MeanDimBackward : public Node<T> {
    std::vector<int64_t> input_shape_;
    int64_t dim_;
    int64_t dim_size_;
public:
    MeanDimBackward(const Tensor<T>& x, int64_t dim)
        : Node<T>(), input_shape_(x.shape()), dim_(dim),
          dim_size_(x.shape()[static_cast<size_t>(dim)])
    {
        this->next_edges.reserve(1);
        this->next_edges.emplace_back(Node<T>::get_next_edge(x));
    }

    std::vector<Tensor<T>> apply(const Tensor<T>& propagated_grad) override
    {
        const int64_t in_rank  = static_cast<int64_t>(input_shape_.size());
        const int64_t in_numel = Tensor<T>::compute_numel(input_shape_);
        const auto in_strides  = Tensor<T>::compute_contiguous_strides(input_shape_);

        std::vector<int64_t> out_shape;
        out_shape.reserve(static_cast<size_t>(in_rank - 1));
        for (int64_t d = 0; d < in_rank; ++d)
            if (d != dim_) out_shape.push_back(input_shape_[static_cast<size_t>(d)]);

        const auto out_strides = Tensor<T>::compute_contiguous_strides(out_shape);

        std::vector<T> grad_in(static_cast<size_t>(in_numel));
        for (int64_t flat = 0; flat < in_numel; ++flat) {
            int64_t remaining = flat;
            int64_t out_flat  = 0;
            int64_t out_d     = 0;
            for (int64_t d = 0; d < in_rank; ++d) {
                const int64_t idx = remaining / in_strides[static_cast<size_t>(d)];
                remaining        %= in_strides[static_cast<size_t>(d)];
                if (d != dim_) {
                    out_flat += idx * out_strides[static_cast<size_t>(out_d)];
                    ++out_d;
                }
            }
            grad_in[static_cast<size_t>(flat)] =
                propagated_grad.data()[static_cast<size_t>(out_flat)] / static_cast<T>(dim_size_);
        }
        return {Tensor<T>::from_operation_result(input_shape_, std::move(grad_in), false, nullptr)};
    }
};

// ----- max() backward -----
// Forward: scalar = maximum element.
// Backward: only the first element that achieved the maximum gets the gradient;
//           all others receive zero (ties broken by first occurrence).
template <typename T>
class MaxBackward : public Node<T> {
    std::vector<int64_t> input_shape_;
    int64_t argmax_;
public:
    MaxBackward(const Tensor<T>& x, int64_t argmax)
        : Node<T>(), input_shape_(x.shape()), argmax_(argmax)
    {
        this->next_edges.reserve(1);
        this->next_edges.emplace_back(Node<T>::get_next_edge(x));
    }

    std::vector<Tensor<T>> apply(const Tensor<T>& propagated_grad) override
    {
        const int64_t numel = Tensor<T>::compute_numel(input_shape_);
        std::vector<T> out(static_cast<size_t>(numel), T{0});
        out[static_cast<size_t>(argmax_)] = propagated_grad.data()[0];
        return {Tensor<T>::from_operation_result(input_shape_, std::move(out), false, nullptr)};
    }
};

// ----- min() backward -----
// Forward: scalar = minimum element.
// Backward: only the first element that achieved the minimum gets the gradient;
//           all others receive zero (ties broken by first occurrence).
template <typename T>
class MinBackward : public Node<T> {
    std::vector<int64_t> input_shape_;
    int64_t argmin_;
public:
    MinBackward(const Tensor<T>& x, int64_t argmin)
        : Node<T>(), input_shape_(x.shape()), argmin_(argmin)
    {
        this->next_edges.reserve(1);
        this->next_edges.emplace_back(Node<T>::get_next_edge(x));
    }

    std::vector<Tensor<T>> apply(const Tensor<T>& propagated_grad) override
    {
        const int64_t numel = Tensor<T>::compute_numel(input_shape_);
        std::vector<T> out(static_cast<size_t>(numel), T{0});
        out[static_cast<size_t>(argmin_)] = propagated_grad.data()[0];
        return {Tensor<T>::from_operation_result(input_shape_, std::move(out), false, nullptr)};
    }
};

} // namespace autograd
