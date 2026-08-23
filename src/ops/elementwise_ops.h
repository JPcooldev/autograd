#pragma once

#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>

#include "../tensor/tensor.h"
#include "../autograd/grad_context.h"
#include "../autograd/backward_ops/backward_elementwise_ops.h"

namespace ops {

template <typename T>
void check_same_shapes(const tensor::Tensor<T>& x, const tensor::Tensor<T>& y, const std::string& op_name) {
    if (x.shape() != y.shape()) {
        throw std::invalid_argument(op_name + " requires tensors with matching shapes");
    }
}

// ----- basic operations -----

template <typename T>
tensor::Tensor<T> add(const tensor::Tensor<T>& x, const tensor::Tensor<T>& y) 
{   
    check_same_shapes(x, y, "add");

    // create data buffer
    std::vector<T> storage(static_cast<size_t>(x.numel()));
    // fill data buffer with the result of the addition
    for (size_t index = 0; index < storage.size(); ++index)
        storage[index] = x.data()[index] + y.data()[index];

    const bool requires_grad = autograd::is_grad_enabled() && (x.requires_grad() || y.requires_grad());
    std::shared_ptr<autograd::Node<T>> grad_fn = nullptr;
    if (requires_grad)
        grad_fn = std::make_shared<autograd::AddBackward<T>>(x, y);
    return tensor::Tensor<T>::from_operation_result(
        x.shape(), std::move(storage), requires_grad, std::move(grad_fn)
    );
}

template <typename T>
tensor::Tensor<T> neg(const tensor::Tensor<T>& x) {
    std::vector<T> storage(static_cast<size_t>(x.numel()));
    for (size_t index = 0; index < storage.size(); ++index)
        storage[index] = static_cast<T>(-1) * x.data()[index];

    const bool requires_grad = autograd::is_grad_enabled() && x.requires_grad();
    std::shared_ptr<autograd::Node<T>> grad_fn = nullptr;
    if (requires_grad)
        grad_fn = std::make_shared<autograd::NegationBackward<T>>(x);
    return tensor::Tensor<T>::from_operation_result(
        x.shape(), std::move(storage), requires_grad, std::move(grad_fn)
    );
}

template <typename T>
tensor::Tensor<T> subtract(const tensor::Tensor<T>& x, const tensor::Tensor<T>& y) {
    check_same_shapes(x, y, "subtract");

    std::vector<T> storage(static_cast<size_t>(x.numel()));
    for (size_t index = 0; index < storage.size(); ++index)
        storage[index] = x.data()[index] - y.data()[index];

    const bool requires_grad = autograd::is_grad_enabled() && (x.requires_grad() || y.requires_grad());
    std::shared_ptr<autograd::Node<T>> grad_fn = nullptr;
    if (requires_grad)
        grad_fn = std::make_shared<autograd::SubtractBackward<T>>(x, y);

    return tensor::Tensor<T>::from_operation_result(
        x.shape(), std::move(storage), requires_grad, std::move(grad_fn)
    );
}

template <typename T>
tensor::Tensor<T> multiply(const tensor::Tensor<T>& x, const tensor::Tensor<T>& y) {
    check_same_shapes(x, y, "multiply");

    std::vector<T> storage(static_cast<size_t>(x.numel()));
    for (size_t index = 0; index < storage.size(); ++index)
        storage[index] = x.data()[index] * y.data()[index];

    const bool requires_grad = autograd::is_grad_enabled() && (x.requires_grad() || y.requires_grad());
    std::shared_ptr<autograd::Node<T>> grad_fn = nullptr;
    if (requires_grad)
        grad_fn = std::make_shared<autograd::MultiplyBackward<T>>(x, y);
    return tensor::Tensor<T>::from_operation_result(
        x.shape(), std::move(storage), requires_grad, std::move(grad_fn)
    );
}

template <typename T>
tensor::Tensor<T> divide(const tensor::Tensor<T>& x, const tensor::Tensor<T>& y) {
    check_same_shapes(x, y, "divide");

    std::vector<T> storage(static_cast<size_t>(x.numel()));
    for (size_t index = 0; index < storage.size(); ++index) {
        if (y.data()[index] == static_cast<T>(0)) 
            throw std::runtime_error("division by zero");
        storage[index] = x.data()[index] / y.data()[index];
    }

    const bool requires_grad = autograd::is_grad_enabled() && (x.requires_grad() || y.requires_grad());
    std::shared_ptr<autograd::Node<T>> grad_fn = nullptr;
    if (requires_grad)
        grad_fn = std::make_shared<autograd::DivideBackward<T>>(x, y);
    return tensor::Tensor<T>::from_operation_result(
        x.shape(), std::move(storage), requires_grad, std::move(grad_fn)
    );
}

template <typename T, typename U>
tensor::Tensor<T> power(const tensor::Tensor<T>& x, const U exponent) {
    std::vector<T> storage(static_cast<size_t>(x.numel()));
    for (size_t index = 0; index < storage.size(); ++index)
        storage[index] = static_cast<T>(std::pow(x.data()[index], static_cast<T>(exponent)));

    const bool requires_grad = autograd::is_grad_enabled() && x.requires_grad();
    std::shared_ptr<autograd::Node<T>> grad_fn = nullptr;
    if (requires_grad)
        grad_fn = std::make_shared<autograd::PowerBackward<T>>(x, static_cast<T>(exponent));
    return tensor::Tensor<T>::from_operation_result(
        x.shape(), std::move(storage), requires_grad, std::move(grad_fn)
    );
}

template <typename T>
tensor::Tensor<T> abs(const tensor::Tensor<T>& x) {
    std::vector<T> storage(static_cast<size_t>(x.numel()));
    for (size_t index = 0; index < storage.size(); ++index)
        storage[index] = static_cast<T>(std::abs(x.data()[index]));

    const bool requires_grad = autograd::is_grad_enabled() && x.requires_grad();
    std::shared_ptr<autograd::Node<T>> grad_fn = nullptr;
    if (requires_grad)
        grad_fn = std::make_shared<autograd::AbsBackward<T>>(x);
    return tensor::Tensor<T>::from_operation_result(
        x.shape(), std::move(storage), requires_grad, std::move(grad_fn)
    );
}

// ----- exp and log -----
template <typename T>
tensor::Tensor<T> exp(const tensor::Tensor<T>& x) {
    std::vector<T> storage(static_cast<size_t>(x.numel()));
    for (size_t index = 0; index < storage.size(); ++index)
        storage[index] = static_cast<T>(std::exp(x.data()[index]));

    const bool requires_grad = autograd::is_grad_enabled() && x.requires_grad();
    std::shared_ptr<autograd::Node<T>> grad_fn = nullptr;
    if (requires_grad)
        grad_fn = std::make_shared<autograd::ExpBackward<T>>(x);
    return tensor::Tensor<T>::from_operation_result(
        x.shape(), std::move(storage), requires_grad, std::move(grad_fn)
    );
}

template <typename T>
tensor::Tensor<T> log(const tensor::Tensor<T>& x) {
    std::vector<T> storage(static_cast<size_t>(x.numel()));
    for (size_t index = 0; index < storage.size(); ++index)
        storage[index] = static_cast<T>(std::log(x.data()[index]));

    const bool requires_grad = autograd::is_grad_enabled() && x.requires_grad();
    std::shared_ptr<autograd::Node<T>> grad_fn = nullptr;
    if (requires_grad)
        grad_fn = std::make_shared<autograd::LogBackward<T>>(x);
    return tensor::Tensor<T>::from_operation_result(
        x.shape(), std::move(storage), requires_grad, std::move(grad_fn)
    );
}

// ----- trigonometric functions -----
template <typename T>
tensor::Tensor<T> sin(const tensor::Tensor<T>& x) {
    std::vector<T> storage(static_cast<size_t>(x.numel()));
    for (size_t index = 0; index < storage.size(); ++index)
        storage[index] = static_cast<T>(std::sin(x.data()[index]));

    const bool requires_grad = autograd::is_grad_enabled() && x.requires_grad();
    std::shared_ptr<autograd::Node<T>> grad_fn = nullptr;
    if (requires_grad)
        grad_fn = std::make_shared<autograd::SinBackward<T>>(x);
    return tensor::Tensor<T>::from_operation_result(
        x.shape(), std::move(storage), requires_grad, std::move(grad_fn)
    );
}

template <typename T>
tensor::Tensor<T> cos(const tensor::Tensor<T>& x) {
    std::vector<T> storage(static_cast<size_t>(x.numel()));
    for (size_t index = 0; index < storage.size(); ++index)
        storage[index] = static_cast<T>(std::cos(x.data()[index]));

    const bool requires_grad = autograd::is_grad_enabled() && x.requires_grad();
    std::shared_ptr<autograd::Node<T>> grad_fn = nullptr;
    if (requires_grad)
        grad_fn = std::make_shared<autograd::CosBackward<T>>(x);
    return tensor::Tensor<T>::from_operation_result(
        x.shape(), std::move(storage), requires_grad, std::move(grad_fn)
    );
}

template <typename T>
tensor::Tensor<T> tan(const tensor::Tensor<T>& x) {
    std::vector<T> storage(static_cast<size_t>(x.numel()));
    for (size_t index = 0; index < storage.size(); ++index)
        storage[index] = static_cast<T>(std::tan(x.data()[index]));

    const bool requires_grad = autograd::is_grad_enabled() && x.requires_grad();
    std::shared_ptr<autograd::Node<T>> grad_fn = nullptr;
    if (requires_grad)
        grad_fn = std::make_shared<autograd::TanBackward<T>>(x);
    return tensor::Tensor<T>::from_operation_result(
        x.shape(), std::move(storage), requires_grad, std::move(grad_fn)
    );
}

// ----- hyperbolic functions -----

// sinh(x) = (exp(x) - exp(-x)) / 2
template <typename T>
tensor::Tensor<T> sinh(const tensor::Tensor<T>& x) {
    std::vector<T> storage(static_cast<size_t>(x.numel()));
    for (size_t i = 0; i < storage.size(); ++i)
        storage[i] = static_cast<T>(std::sinh(x.data()[i]));

    const bool requires_grad = autograd::is_grad_enabled() && x.requires_grad();
    std::shared_ptr<autograd::Node<T>> grad_fn = nullptr;
    if (requires_grad)
        grad_fn = std::make_shared<autograd::SinhBackward<T>>(x);
    return tensor::Tensor<T>::from_operation_result(
        x.shape(), std::move(storage), requires_grad, std::move(grad_fn)
    );
}

// cosh(x) = (exp(x) + exp(-x)) / 2
template <typename T>
tensor::Tensor<T> cosh(const tensor::Tensor<T>& x) {
    std::vector<T> storage(static_cast<size_t>(x.numel()));
    for (size_t i = 0; i < storage.size(); ++i)
        storage[i] = static_cast<T>(std::cosh(x.data()[i]));

    const bool requires_grad = autograd::is_grad_enabled() && x.requires_grad();
    std::shared_ptr<autograd::Node<T>> grad_fn = nullptr;
    if (requires_grad)
        grad_fn = std::make_shared<autograd::CoshBackward<T>>(x);
    return tensor::Tensor<T>::from_operation_result(
        x.shape(), std::move(storage), requires_grad, std::move(grad_fn)
    );
}

// tanh(x) = sinh(x) / cosh(x)
template <typename T>
tensor::Tensor<T> tanh(const tensor::Tensor<T>& x) {
    std::vector<T> storage(static_cast<size_t>(x.numel()));
    for (size_t i = 0; i < storage.size(); ++i)
        storage[i] = static_cast<T>(std::tanh(x.data()[i]));

    const bool requires_grad = autograd::is_grad_enabled() && x.requires_grad();
    std::shared_ptr<autograd::Node<T>> grad_fn = nullptr;
    if (requires_grad)
        grad_fn = std::make_shared<autograd::TanhBackward<T>>(x);
    return tensor::Tensor<T>::from_operation_result(
        x.shape(), std::move(storage), requires_grad, std::move(grad_fn)
    );
}

// ----- other non-linear functions -----

template <typename T>
inline T sigmoid_scalar(T x) {
    return static_cast<T>(1) / (static_cast<T>(1) + static_cast<T>(std::exp(-x)));
}

// sigmoid(x) = 1 / (1 + exp(-x))
template <typename T>
tensor::Tensor<T> sigmoid(const tensor::Tensor<T>& x) {
    std::vector<T> storage(static_cast<size_t>(x.numel()));
    for (size_t i = 0; i < storage.size(); ++i)
        storage[i] = sigmoid_scalar(x.data()[i]);

    const bool requires_grad = autograd::is_grad_enabled() && x.requires_grad();
    std::shared_ptr<autograd::Node<T>> grad_fn = nullptr;
    if (requires_grad)
        grad_fn = std::make_shared<autograd::SigmoidBackward<T>>(x);
    return tensor::Tensor<T>::from_operation_result(
        x.shape(), std::move(storage), requires_grad, std::move(grad_fn)
    );
}

// relu(x) = max(0, x)
template <typename T>
tensor::Tensor<T> relu(const tensor::Tensor<T>& x) {
    std::vector<T> storage(static_cast<size_t>(x.numel()));
    for (size_t i = 0; i < storage.size(); ++i)
        storage[i] = x.data()[i] > static_cast<T>(0) ? x.data()[i] : static_cast<T>(0);

    const bool requires_grad = autograd::is_grad_enabled() && x.requires_grad();
    std::shared_ptr<autograd::Node<T>> grad_fn = nullptr;
    if (requires_grad)
        grad_fn = std::make_shared<autograd::ReluBackward<T>>(x);
    return tensor::Tensor<T>::from_operation_result(
        x.shape(), std::move(storage), requires_grad, std::move(grad_fn));
}

// SiLU (Sigmoid Linear Unit)
// SiLU(x) = x * sigmoid(x)
template <typename T>
tensor::Tensor<T> silu(const tensor::Tensor<T>& x) {
    std::vector<T> storage(static_cast<size_t>(x.numel()));
    for (size_t i = 0; i < storage.size(); ++i)
        storage[i] = x.data()[i] * sigmoid_scalar(x.data()[i]);

    const bool requires_grad = autograd::is_grad_enabled() && x.requires_grad();
    std::shared_ptr<autograd::Node<T>> grad_fn = nullptr;
    if (requires_grad)
        grad_fn = std::make_shared<autograd::SiLUBackward<T>>(x);
    return tensor::Tensor<T>::from_operation_result(
        x.shape(), std::move(storage), requires_grad, std::move(grad_fn)
    );
}

// GELU (Gaussian Error Linear Unit)
// Approximated by:
// GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
template <typename T>
tensor::Tensor<T> gelu(const tensor::Tensor<T>& x) {
    const T k = static_cast<T>(std::sqrt(2.0 / M_PI));  // sqrt(2/pi)
    constexpr T c = static_cast<T>(0.044715);

    std::vector<T> storage(static_cast<size_t>(x.numel()));
    for (size_t i = 0; i < storage.size(); ++i) {
        const T xi = x.data()[i];
        const T u  = k * (xi + c * static_cast<T>(std::pow(xi, 3)));
        storage[i] = static_cast<T>(0.5) * xi * (static_cast<T>(1) + static_cast<T>(std::tanh(u)));
    }

    const bool requires_grad = autograd::is_grad_enabled() && x.requires_grad();
    std::shared_ptr<autograd::Node<T>> grad_fn = nullptr;
    if (requires_grad)
        grad_fn = std::make_shared<autograd::GELUBackward<T>>(x);
    return tensor::Tensor<T>::from_operation_result(
        x.shape(), std::move(storage), requires_grad, std::move(grad_fn)
    );
}

} // namespace ops