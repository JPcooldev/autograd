#pragma once

#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "../tensor/tensor.h"
#include "../autograd/grad_context.h"
#include "../autograd/backward_ops/backward_elementwise_ops.h"
#include "shape_ops.h"

namespace ops {

/**
 * Check that two tensors have the same shape. Compares `x.shape()` and
 * `y.shape()` and throws if they differ.
 *
 * @param x First tensor.
 * @param y Second tensor.
 * @param op_name Name of the calling op, used in the error message.
 *
 * @throws std::invalid_argument if the shapes do not match.
 */
template <typename T>
void check_same_shapes(
    const tensor::Tensor<T>& x,
    const tensor::Tensor<T>& y,
    const std::string& op_name) {
    if (x.shape() != y.shape())
        throw std::invalid_argument(op_name + " requires tensors with matching shapes");
}

// ----- basic operations -----

/**
 * Add two tensors elementwise. Packs both operands, then writes `x[i] + y[i]`
 * into a new buffer and attaches `AddBackward` when grad is enabled.
 *
 * @param x Left operand.
 * @param y Right operand.
 * @return A tensor with the same shape as `x` (and `y`).
 *
 * @throws std::invalid_argument if the tensors have different shapes.
 */
template <typename T>
tensor::Tensor<T> add(const tensor::Tensor<T>& x, const tensor::Tensor<T>& y) {
    check_same_shapes(x, y, "add");
    const auto xc = x.contiguous();
    const auto yc = y.contiguous();

    std::vector<T> storage(static_cast<size_t>(xc.numel()));
    for (size_t index = 0; index < storage.size(); ++index)
        storage[index] = xc.data()[index] + yc.data()[index];

    const bool requires_grad = autograd::is_grad_enabled() && (x.requires_grad() || y.requires_grad());
    std::shared_ptr<autograd::Node<T>> grad_fn = nullptr;
    if (requires_grad)
        grad_fn = std::make_shared<autograd::AddBackward<T>>(xc, yc);
    return tensor::Tensor<T>::from_operation_result(
        x.shape(), std::move(storage), requires_grad, std::move(grad_fn));
}

/**
 * Negate a tensor elementwise. Packs `x`, then writes `-x[i]` into a new
 * buffer and attaches `NegationBackward` when grad is enabled.
 *
 * @param x The tensor to negate.
 * @return A tensor with the same shape as `x`.
 */
template <typename T>
tensor::Tensor<T> neg(const tensor::Tensor<T>& x) {
    const auto xc = x.contiguous();
    std::vector<T> storage(static_cast<size_t>(xc.numel()));
    for (size_t index = 0; index < storage.size(); ++index)
        storage[index] = static_cast<T>(-1) * xc.data()[index];

    const bool requires_grad = autograd::is_grad_enabled() && x.requires_grad();
    std::shared_ptr<autograd::Node<T>> grad_fn = nullptr;
    if (requires_grad)
        grad_fn = std::make_shared<autograd::NegationBackward<T>>(xc);
    return tensor::Tensor<T>::from_operation_result(
        x.shape(), std::move(storage), requires_grad, std::move(grad_fn));
}

/**
 * Subtract two tensors elementwise. Packs both operands, then writes
 * `x[i] - y[i]` into a new buffer and attaches `SubtractBackward` when grad is
 * enabled.
 *
 * @param x Left operand.
 * @param y Right operand.
 * @return A tensor with the same shape as `x` (and `y`).
 *
 * @throws std::invalid_argument if the tensors have different shapes.
 */
template <typename T>
tensor::Tensor<T> subtract(const tensor::Tensor<T>& x, const tensor::Tensor<T>& y) {
    check_same_shapes(x, y, "subtract");
    const auto xc = x.contiguous();
    const auto yc = y.contiguous();

    std::vector<T> storage(static_cast<size_t>(xc.numel()));
    for (size_t index = 0; index < storage.size(); ++index)
        storage[index] = xc.data()[index] - yc.data()[index];

    const bool requires_grad = autograd::is_grad_enabled() && (x.requires_grad() || y.requires_grad());
    std::shared_ptr<autograd::Node<T>> grad_fn = nullptr;
    if (requires_grad)
        grad_fn = std::make_shared<autograd::SubtractBackward<T>>(xc, yc);
    return tensor::Tensor<T>::from_operation_result(
        x.shape(), std::move(storage), requires_grad, std::move(grad_fn));
}

/**
 * Multiply two tensors elementwise. Packs both operands, then writes
 * `x[i] * y[i]` into a new buffer and attaches `MultiplyBackward` when grad is
 * enabled.
 *
 * @param x Left operand.
 * @param y Right operand.
 * @return A tensor with the same shape as `x` (and `y`).
 *
 * @throws std::invalid_argument if the tensors have different shapes.
 */
template <typename T>
tensor::Tensor<T> multiply(const tensor::Tensor<T>& x, const tensor::Tensor<T>& y) {
    check_same_shapes(x, y, "multiply");
    const auto xc = x.contiguous();
    const auto yc = y.contiguous();

    std::vector<T> storage(static_cast<size_t>(xc.numel()));
    for (size_t index = 0; index < storage.size(); ++index)
        storage[index] = xc.data()[index] * yc.data()[index];

    const bool requires_grad = autograd::is_grad_enabled() && (x.requires_grad() || y.requires_grad());
    std::shared_ptr<autograd::Node<T>> grad_fn = nullptr;
    if (requires_grad)
        grad_fn = std::make_shared<autograd::MultiplyBackward<T>>(xc, yc);
    return tensor::Tensor<T>::from_operation_result(
        x.shape(), std::move(storage), requires_grad, std::move(grad_fn));
}

/**
 * Divide two tensors elementwise. Packs both operands, then writes
 * `x[i] / y[i]` into a new buffer and attaches `DivideBackward` when grad is
 * enabled.
 *
 * @param x Numerator.
 * @param y Denominator.
 * @return A tensor with the same shape as `x` (and `y`).
 *
 * @throws std::invalid_argument if the tensors have different shapes.
 * @throws std::runtime_error if any element of `y` is zero.
 */
template <typename T>
tensor::Tensor<T> divide(const tensor::Tensor<T>& x, const tensor::Tensor<T>& y) {
    check_same_shapes(x, y, "divide");
    const auto xc = x.contiguous();
    const auto yc = y.contiguous();

    std::vector<T> storage(static_cast<size_t>(xc.numel()));
    for (size_t index = 0; index < storage.size(); ++index) {
        if (yc.data()[index] == static_cast<T>(0))
            throw std::runtime_error("division by zero");
        storage[index] = xc.data()[index] / yc.data()[index];
    }

    const bool requires_grad = autograd::is_grad_enabled() && (x.requires_grad() || y.requires_grad());
    std::shared_ptr<autograd::Node<T>> grad_fn = nullptr;
    if (requires_grad)
        grad_fn = std::make_shared<autograd::DivideBackward<T>>(xc, yc);
    return tensor::Tensor<T>::from_operation_result(
        x.shape(), std::move(storage), requires_grad, std::move(grad_fn));
}

/**
 * Raise each element of a tensor to a scalar power. Packs `x`, then writes
 * `pow(x[i], exponent)` into a new buffer and attaches `PowerBackward` when
 * grad is enabled.
 *
 * @param x The base tensor.
 * @param exponent The scalar exponent (cast to `T`).
 * @return A tensor with the same shape as `x`.
 */
template <typename T, typename U>
tensor::Tensor<T> power(const tensor::Tensor<T>& x, const U exponent) {
    const auto xc = x.contiguous();
    std::vector<T> storage(static_cast<size_t>(xc.numel()));
    for (size_t index = 0; index < storage.size(); ++index)
        storage[index] = static_cast<T>(std::pow(xc.data()[index], static_cast<T>(exponent)));

    const bool requires_grad = autograd::is_grad_enabled() && x.requires_grad();
    std::shared_ptr<autograd::Node<T>> grad_fn = nullptr;
    if (requires_grad)
        grad_fn = std::make_shared<autograd::PowerBackward<T>>(xc, static_cast<T>(exponent));
    return tensor::Tensor<T>::from_operation_result(
        x.shape(), std::move(storage), requires_grad, std::move(grad_fn));
}

/**
 * Take the absolute value of each element. Packs `x`, then writes `abs(x[i])`
 * into a new buffer and attaches `AbsBackward` when grad is enabled.
 *
 * @param x The tensor to take the absolute value of.
 * @return A tensor with the same shape as `x`.
 */
template <typename T>
tensor::Tensor<T> abs(const tensor::Tensor<T>& x) {
    const auto xc = x.contiguous();
    std::vector<T> storage(static_cast<size_t>(xc.numel()));
    for (size_t index = 0; index < storage.size(); ++index)
        storage[index] = static_cast<T>(std::abs(xc.data()[index]));

    const bool requires_grad = autograd::is_grad_enabled() && x.requires_grad();
    std::shared_ptr<autograd::Node<T>> grad_fn = nullptr;
    if (requires_grad)
        grad_fn = std::make_shared<autograd::AbsBackward<T>>(xc);
    return tensor::Tensor<T>::from_operation_result(
        x.shape(), std::move(storage), requires_grad, std::move(grad_fn));
}

/**
 * Take the square root of each element. Packs `x`, then writes `sqrt(x[i])`
 * into a new buffer and attaches `SqrtBackward` when grad is enabled.
 *
 * @param x The tensor to take the square root of.
 * @return A tensor with the same shape as `x`.
 */
template <typename T>
tensor::Tensor<T> sqrt(const tensor::Tensor<T>& x) {
    const auto xc = x.contiguous();
    std::vector<T> storage(static_cast<size_t>(xc.numel()));
    for (size_t index = 0; index < storage.size(); ++index)
        storage[index] = static_cast<T>(std::sqrt(xc.data()[index]));

    const bool requires_grad = autograd::is_grad_enabled() && x.requires_grad();
    std::shared_ptr<autograd::Node<T>> grad_fn = nullptr;
    if (requires_grad)
        grad_fn = std::make_shared<autograd::SqrtBackward<T>>(xc);
    return tensor::Tensor<T>::from_operation_result(
        x.shape(), std::move(storage), requires_grad, std::move(grad_fn));
}

// ----- exp and log -----

/**
 * Apply `exp` to each element. Packs `x`, then writes `exp(x[i])` into a new
 * buffer and attaches `ExpBackward` when grad is enabled.
 *
 * @param x The tensor to exponentiate.
 * @return A tensor with the same shape as `x`.
 */
template <typename T>
tensor::Tensor<T> exp(const tensor::Tensor<T>& x) {
    const auto xc = x.contiguous();
    std::vector<T> storage(static_cast<size_t>(xc.numel()));
    for (size_t index = 0; index < storage.size(); ++index)
        storage[index] = static_cast<T>(std::exp(xc.data()[index]));

    const bool requires_grad = autograd::is_grad_enabled() && x.requires_grad();
    std::shared_ptr<autograd::Node<T>> grad_fn = nullptr;
    if (requires_grad)
        grad_fn = std::make_shared<autograd::ExpBackward<T>>(xc);
    return tensor::Tensor<T>::from_operation_result(
        x.shape(), std::move(storage), requires_grad, std::move(grad_fn));
}

/**
 * Apply the natural logarithm to each element. Packs `x`, then writes
 * `log(x[i])` into a new buffer and attaches `LogBackward` when grad is enabled.
 *
 * @param x The tensor to take the log of.
 * @return A tensor with the same shape as `x`.
 *
 * @throws std::runtime_error if any element of `x` is not strictly positive.
 */
template <typename T>
tensor::Tensor<T> log(const tensor::Tensor<T>& x) {
    const auto xc = x.contiguous();
    std::vector<T> storage(static_cast<size_t>(xc.numel()));
    for (size_t index = 0; index < storage.size(); ++index) {
        const auto x_i = xc.data()[index];
        if (x_i <= static_cast<T>(0))
            throw std::runtime_error("log of non-positive value");
        storage[index] = static_cast<T>(std::log(x_i));
    }

    const bool requires_grad = autograd::is_grad_enabled() && x.requires_grad();
    std::shared_ptr<autograd::Node<T>> grad_fn = nullptr;
    if (requires_grad)
        grad_fn = std::make_shared<autograd::LogBackward<T>>(xc);
    return tensor::Tensor<T>::from_operation_result(
        x.shape(), std::move(storage), requires_grad, std::move(grad_fn));
}

// ----- trigonometric functions -----

/**
 * Apply `sin` to each element. Packs `x`, then writes `sin(x[i])` into a new
 * buffer and attaches `SinBackward` when grad is enabled.
 *
 * @param x The tensor to apply sine to.
 * @return A tensor with the same shape as `x`.
 */
template <typename T>
tensor::Tensor<T> sin(const tensor::Tensor<T>& x) {
    const auto xc = x.contiguous();
    std::vector<T> storage(static_cast<size_t>(xc.numel()));
    for (size_t index = 0; index < storage.size(); ++index)
        storage[index] = static_cast<T>(std::sin(xc.data()[index]));

    const bool requires_grad = autograd::is_grad_enabled() && x.requires_grad();
    std::shared_ptr<autograd::Node<T>> grad_fn = nullptr;
    if (requires_grad)
        grad_fn = std::make_shared<autograd::SinBackward<T>>(xc);
    return tensor::Tensor<T>::from_operation_result(
        x.shape(), std::move(storage), requires_grad, std::move(grad_fn));
}

/**
 * Apply `cos` to each element. Packs `x`, then writes `cos(x[i])` into a new
 * buffer and attaches `CosBackward` when grad is enabled.
 *
 * @param x The tensor to apply cosine to.
 * @return A tensor with the same shape as `x`.
 */
template <typename T>
tensor::Tensor<T> cos(const tensor::Tensor<T>& x) {
    const auto xc = x.contiguous();
    std::vector<T> storage(static_cast<size_t>(xc.numel()));
    for (size_t index = 0; index < storage.size(); ++index)
        storage[index] = static_cast<T>(std::cos(xc.data()[index]));

    const bool requires_grad = autograd::is_grad_enabled() && x.requires_grad();
    std::shared_ptr<autograd::Node<T>> grad_fn = nullptr;
    if (requires_grad)
        grad_fn = std::make_shared<autograd::CosBackward<T>>(xc);
    return tensor::Tensor<T>::from_operation_result(
        x.shape(), std::move(storage), requires_grad, std::move(grad_fn));
}

/**
 * Apply `tan` to each element. Packs `x`, then writes `tan(x[i])` into a new
 * buffer and attaches `TanBackward` when grad is enabled.
 *
 * @param x The tensor to apply tangent to.
 * @return A tensor with the same shape as `x`.
 */
template <typename T>
tensor::Tensor<T> tan(const tensor::Tensor<T>& x) {
    const auto xc = x.contiguous();
    std::vector<T> storage(static_cast<size_t>(xc.numel()));
    for (size_t index = 0; index < storage.size(); ++index)
        storage[index] = static_cast<T>(std::tan(xc.data()[index]));

    const bool requires_grad = autograd::is_grad_enabled() && x.requires_grad();
    std::shared_ptr<autograd::Node<T>> grad_fn = nullptr;
    if (requires_grad)
        grad_fn = std::make_shared<autograd::TanBackward<T>>(xc);
    return tensor::Tensor<T>::from_operation_result(
        x.shape(), std::move(storage), requires_grad, std::move(grad_fn));
}

// ----- hyperbolic functions -----

/**
 * Apply `sinh` to each element. Packs `x`, then writes `sinh(x[i])` into a
 * new buffer and attaches `SinhBackward` when grad is enabled.
 *
 * @param x The tensor to apply hyperbolic sine to.
 * @return A tensor with the same shape as `x`.
 */
template <typename T>
tensor::Tensor<T> sinh(const tensor::Tensor<T>& x) {
    const auto xc = x.contiguous();
    std::vector<T> storage(static_cast<size_t>(xc.numel()));
    for (size_t i = 0; i < storage.size(); ++i)
        storage[i] = static_cast<T>(std::sinh(xc.data()[i]));

    const bool requires_grad = autograd::is_grad_enabled() && x.requires_grad();
    std::shared_ptr<autograd::Node<T>> grad_fn = nullptr;
    if (requires_grad)
        grad_fn = std::make_shared<autograd::SinhBackward<T>>(xc);
    return tensor::Tensor<T>::from_operation_result(
        x.shape(), std::move(storage), requires_grad, std::move(grad_fn));
}

/**
 * Apply `cosh` to each element. Packs `x`, then writes `cosh(x[i])` into a
 * new buffer and attaches `CoshBackward` when grad is enabled.
 *
 * @param x The tensor to apply hyperbolic cosine to.
 * @return A tensor with the same shape as `x`.
 */
template <typename T>
tensor::Tensor<T> cosh(const tensor::Tensor<T>& x) {
    const auto xc = x.contiguous();
    std::vector<T> storage(static_cast<size_t>(xc.numel()));
    for (size_t i = 0; i < storage.size(); ++i)
        storage[i] = static_cast<T>(std::cosh(xc.data()[i]));

    const bool requires_grad = autograd::is_grad_enabled() && x.requires_grad();
    std::shared_ptr<autograd::Node<T>> grad_fn = nullptr;
    if (requires_grad)
        grad_fn = std::make_shared<autograd::CoshBackward<T>>(xc);
    return tensor::Tensor<T>::from_operation_result(
        x.shape(), std::move(storage), requires_grad, std::move(grad_fn));
}

/**
 * Apply `tanh` to each element. Packs `x`, then writes `tanh(x[i])` into a
 * new buffer and attaches `TanhBackward` when grad is enabled.
 *
 * @param x The tensor to apply hyperbolic tangent to.
 * @return A tensor with the same shape as `x`.
 */
template <typename T>
tensor::Tensor<T> tanh(const tensor::Tensor<T>& x) {
    const auto xc = x.contiguous();
    std::vector<T> storage(static_cast<size_t>(xc.numel()));
    for (size_t i = 0; i < storage.size(); ++i)
        storage[i] = static_cast<T>(std::tanh(xc.data()[i]));

    const bool requires_grad = autograd::is_grad_enabled() && x.requires_grad();
    std::shared_ptr<autograd::Node<T>> grad_fn = nullptr;
    if (requires_grad)
        grad_fn = std::make_shared<autograd::TanhBackward<T>>(xc);
    return tensor::Tensor<T>::from_operation_result(
        x.shape(), std::move(storage), requires_grad, std::move(grad_fn));
}

// ----- other non-linear functions -----

/**
 * Compute the logistic sigmoid of a scalar. Returns `1 / (1 + exp(-x))`.
 *
 * @param x The scalar input.
 * @return The sigmoid of `x`.
 */
template <typename T>
inline T sigmoid_scalar(T x) {
    return static_cast<T>(1) / (static_cast<T>(1) + static_cast<T>(std::exp(-x)));
}

/**
 * Apply the logistic sigmoid to each element. Packs `x`, then writes
 * `sigmoid(x[i])` into a new buffer and attaches `SigmoidBackward` when grad
 * is enabled.
 *
 * @param x The tensor to apply sigmoid to.
 * @return A tensor with the same shape as `x`.
 */
template <typename T>
tensor::Tensor<T> sigmoid(const tensor::Tensor<T>& x) {
    const auto xc = x.contiguous();
    std::vector<T> storage(static_cast<size_t>(xc.numel()));
    for (size_t i = 0; i < storage.size(); ++i)
        storage[i] = sigmoid_scalar(xc.data()[i]);

    const bool requires_grad = autograd::is_grad_enabled() && x.requires_grad();
    std::shared_ptr<autograd::Node<T>> grad_fn = nullptr;
    if (requires_grad)
        grad_fn = std::make_shared<autograd::SigmoidBackward<T>>(xc);
    return tensor::Tensor<T>::from_operation_result(
        x.shape(), std::move(storage), requires_grad, std::move(grad_fn));
}

/**
 * Apply ReLU to each element. Packs `x`, then writes `max(0, x[i])` into a
 * new buffer and attaches `ReluBackward` when grad is enabled.
 *
 * @param x The tensor to apply ReLU to.
 * @return A tensor with the same shape as `x`.
 */
template <typename T>
tensor::Tensor<T> relu(const tensor::Tensor<T>& x) {
    const auto xc = x.contiguous();
    std::vector<T> storage(static_cast<size_t>(xc.numel()));
    for (size_t i = 0; i < storage.size(); ++i)
        storage[i] = xc.data()[i] > static_cast<T>(0) ? xc.data()[i] : static_cast<T>(0);

    const bool requires_grad = autograd::is_grad_enabled() && x.requires_grad();
    std::shared_ptr<autograd::Node<T>> grad_fn = nullptr;
    if (requires_grad)
        grad_fn = std::make_shared<autograd::ReluBackward<T>>(xc);
    return tensor::Tensor<T>::from_operation_result(
        x.shape(), std::move(storage), requires_grad, std::move(grad_fn));
}

/**
 * Apply SiLU to each element. Packs `x`, then writes `x[i] * sigmoid(x[i])`
 * into a new buffer and attaches `SiLUBackward` when grad is enabled.
 *
 * @param x The tensor to apply SiLU to.
 * @return A tensor with the same shape as `x`.
 */
template <typename T>
tensor::Tensor<T> silu(const tensor::Tensor<T>& x) {
    const auto xc = x.contiguous();
    std::vector<T> storage(static_cast<size_t>(xc.numel()));
    for (size_t i = 0; i < storage.size(); ++i)
        storage[i] = xc.data()[i] * sigmoid_scalar(xc.data()[i]);

    const bool requires_grad = autograd::is_grad_enabled() && x.requires_grad();
    std::shared_ptr<autograd::Node<T>> grad_fn = nullptr;
    if (requires_grad)
        grad_fn = std::make_shared<autograd::SiLUBackward<T>>(xc);
    return tensor::Tensor<T>::from_operation_result(
        x.shape(), std::move(storage), requires_grad, std::move(grad_fn));
}

/**
 * Apply the tanh GELU approximation to each element.
 * Uses `0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))` and attaches
 * `GELUBackward` when grad is enabled.
 *
 * @param x The tensor to apply GELU to.
 * @return A tensor with the same shape as `x`.
 */
template <typename T>
tensor::Tensor<T> gelu(const tensor::Tensor<T>& x) {
    const T k = static_cast<T>(std::sqrt(2.0 / M_PI));
    constexpr T c = static_cast<T>(0.044715);

    const auto xc = x.contiguous();
    std::vector<T> storage(static_cast<size_t>(xc.numel()));
    for (size_t i = 0; i < storage.size(); ++i) {
        const T xi = xc.data()[i];
        const T u  = k * (xi + c * static_cast<T>(std::pow(xi, 3)));
        storage[i] = static_cast<T>(0.5) * xi * (static_cast<T>(1) + static_cast<T>(std::tanh(u)));
    }

    const bool requires_grad = autograd::is_grad_enabled() && x.requires_grad();
    std::shared_ptr<autograd::Node<T>> grad_fn = nullptr;
    if (requires_grad)
        grad_fn = std::make_shared<autograd::GELUBackward<T>>(xc);
    return tensor::Tensor<T>::from_operation_result(
        x.shape(), std::move(storage), requires_grad, std::move(grad_fn));
}

} // namespace ops
