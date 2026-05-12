#pragma once

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

#include "../tensor/tensor.h"
#include "../autograd/grad_context.h"
#include "../autograd/backward_ops/backward_reduction_ops.h"

namespace ops {

// ----- sum (global) -----
// Reduces all elements to a scalar tensor of shape {}.
template <typename T>
tensor::Tensor<T> sum(const tensor::Tensor<T>& x)
{
    if (x.numel() == 0)
        throw std::invalid_argument("sum requires a non-empty tensor");

    T total = T{0};
    for (const T& val : x.data())
        total += val;

    const bool requires_grad = autograd::is_grad_enabled() && x.requires_grad();
    std::shared_ptr<autograd::Node<T>> grad_fn = nullptr;
    if (requires_grad)
        grad_fn = std::make_shared<autograd::SumBackward<T>>(x);

    return tensor::Tensor<T>::from_operation_result(
        {}, std::vector<T>{total}, requires_grad, std::move(grad_fn));
}

// ----- sum along dimension -----
// Reduces along dim; output shape = input shape with dim removed.
// Supports negative dim indices.
template <typename T>
tensor::Tensor<T> sum(const tensor::Tensor<T>& x, int64_t dim)
{
    if (x.rank() == 0)
        throw std::invalid_argument("sum(dim) requires a non-scalar tensor");

    dim = tensor::Tensor<T>::normalize_dimension(dim, x.rank());

    const auto& in_shape = x.shape();
    const int64_t in_rank = x.rank();
    const auto in_strides = tensor::Tensor<T>::compute_contiguous_strides(in_shape);

    std::vector<int64_t> out_shape;
    out_shape.reserve(static_cast<size_t>(in_rank - 1));
    for (int64_t d = 0; d < in_rank; ++d)
        if (d != dim) out_shape.push_back(in_shape[static_cast<size_t>(d)]);

    const int64_t out_numel = tensor::Tensor<T>::compute_numel(out_shape);
    std::vector<T> storage(static_cast<size_t>(out_numel), T{0});

    const auto out_strides = tensor::Tensor<T>::compute_contiguous_strides(out_shape);

    for (int64_t flat = 0; flat < x.numel(); ++flat) {
        int64_t remaining = flat;
        int64_t out_flat = 0;
        int64_t out_d = 0;
        for (int64_t d = 0; d < in_rank; ++d) {
            const int64_t idx = remaining / in_strides[static_cast<size_t>(d)];
            remaining %= in_strides[static_cast<size_t>(d)];
            if (d != dim) {
                out_flat += idx * out_strides[static_cast<size_t>(out_d)];
                ++out_d;
            }
        }
        storage[static_cast<size_t>(out_flat)] += x.data()[static_cast<size_t>(flat)];
    }

    const bool requires_grad = autograd::is_grad_enabled() && x.requires_grad();
    std::shared_ptr<autograd::Node<T>> grad_fn = nullptr;
    if (requires_grad)
        grad_fn = std::make_shared<autograd::SumDimBackward<T>>(x, dim);

    return tensor::Tensor<T>::from_operation_result(
        out_shape, std::move(storage), requires_grad, std::move(grad_fn));
}

// ----- mean (global) -----
// Reduces all elements to a scalar tensor of shape {}.
template <typename T>
tensor::Tensor<T> mean(const tensor::Tensor<T>& x)
{
    if (x.numel() == 0)
        throw std::invalid_argument("mean requires a non-empty tensor");

    T total = T{0};
    for (const T& val : x.data())
        total += val;

    const T result = total / static_cast<T>(x.numel());

    const bool requires_grad = autograd::is_grad_enabled() && x.requires_grad();
    std::shared_ptr<autograd::Node<T>> grad_fn = nullptr;
    if (requires_grad)
        grad_fn = std::make_shared<autograd::MeanBackward<T>>(x);

    return tensor::Tensor<T>::from_operation_result(
        {}, std::vector<T>{result}, requires_grad, std::move(grad_fn));
}

// ----- mean along dimension -----
// Reduces along dim; output shape = input shape with dim removed.
// Supports negative dim indices.
template <typename T>
tensor::Tensor<T> mean(const tensor::Tensor<T>& x, int64_t dim)
{
    if (x.rank() == 0)
        throw std::invalid_argument("mean(dim) requires a non-scalar tensor");

    dim = tensor::Tensor<T>::normalize_dimension(dim, x.rank());

    const auto&   in_shape   = x.shape();
    const int64_t in_rank    = x.rank();
    const int64_t dim_size   = in_shape[static_cast<size_t>(dim)];

    if (dim_size == 0)
        throw std::invalid_argument("mean(dim) cannot reduce over an empty dimension");

    const auto in_strides = tensor::Tensor<T>::compute_contiguous_strides(in_shape);

    std::vector<int64_t> out_shape;
    out_shape.reserve(static_cast<size_t>(in_rank - 1));
    for (int64_t d = 0; d < in_rank; ++d)
        if (d != dim) out_shape.push_back(in_shape[static_cast<size_t>(d)]);

    const int64_t out_numel = tensor::Tensor<T>::compute_numel(out_shape);
    std::vector<T> storage(static_cast<size_t>(out_numel), T{0});

    const auto out_strides = tensor::Tensor<T>::compute_contiguous_strides(out_shape);

    for (int64_t flat = 0; flat < x.numel(); ++flat) {
        int64_t remaining = flat;
        int64_t out_flat  = 0;
        int64_t out_d     = 0;
        for (int64_t d = 0; d < in_rank; ++d) {
            const int64_t idx = remaining / in_strides[static_cast<size_t>(d)];
            remaining        %= in_strides[static_cast<size_t>(d)];
            if (d != dim) {
                out_flat += idx * out_strides[static_cast<size_t>(out_d)];
                ++out_d;
            }
        }
        storage[static_cast<size_t>(out_flat)] += x.data()[static_cast<size_t>(flat)];
    }

    for (T& val : storage)
        val /= static_cast<T>(dim_size);

    const bool requires_grad = autograd::is_grad_enabled() && x.requires_grad();
    std::shared_ptr<autograd::Node<T>> grad_fn = nullptr;
    if (requires_grad)
        grad_fn = std::make_shared<autograd::MeanDimBackward<T>>(x, dim);

    return tensor::Tensor<T>::from_operation_result(
        out_shape, std::move(storage), requires_grad, std::move(grad_fn));
}

// ----- max (global) -----
// Returns the maximum element as a scalar tensor of shape {}.
// Ties are broken by first occurrence.
template <typename T>
tensor::Tensor<T> max(const tensor::Tensor<T>& x)
{
    if (x.numel() == 0)
        throw std::invalid_argument("max requires a non-empty tensor");

    const auto& data = x.data();
    int64_t argmax = 0;
    T max_val = data[0];
    for (int64_t i = 1; i < x.numel(); ++i) {
        if (data[static_cast<size_t>(i)] > max_val) {
            max_val = data[static_cast<size_t>(i)];
            argmax  = i;
        }
    }

    const bool requires_grad = autograd::is_grad_enabled() && x.requires_grad();
    std::shared_ptr<autograd::Node<T>> grad_fn = nullptr;
    if (requires_grad)
        grad_fn = std::make_shared<autograd::MaxBackward<T>>(x, argmax);

    return tensor::Tensor<T>::from_operation_result(
        {}, std::vector<T>{max_val}, requires_grad, std::move(grad_fn));
}

// ----- min (global) -----
// Returns the minimum element as a scalar tensor of shape {}.
// Ties are broken by first occurrence.
template <typename T>
tensor::Tensor<T> min(const tensor::Tensor<T>& x)
{
    if (x.numel() == 0)
        throw std::invalid_argument("min requires a non-empty tensor");

    const auto& data = x.data();
    int64_t argmin = 0;
    T min_val = data[0];
    for (int64_t i = 1; i < x.numel(); ++i) {
        if (data[static_cast<size_t>(i)] < min_val) {
            min_val = data[static_cast<size_t>(i)];
            argmin  = i;
        }
    }

    const bool requires_grad = autograd::is_grad_enabled() && x.requires_grad();
    std::shared_ptr<autograd::Node<T>> grad_fn = nullptr;
    if (requires_grad)
        grad_fn = std::make_shared<autograd::MinBackward<T>>(x, argmin);

    return tensor::Tensor<T>::from_operation_result(
        {}, std::vector<T>{min_val}, requires_grad, std::move(grad_fn));
}

} // namespace ops
