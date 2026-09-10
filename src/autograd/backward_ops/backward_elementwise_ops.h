#pragma once

#include <cmath>
#include <vector>
#include <memory>

#include "../node.h"

namespace autograd {

using tensor::Tensor;

template <typename T>
class AddBackward : public Node<T> {
public:
    /**
     * Construct the backward node for element-wise addition.
     * Aliases `x` and `y` onto `saved_tensors`.
     *
     * @param x Left forward operand.
     * @param y Right forward operand.
     */
    AddBackward(const Tensor<T>& x, const Tensor<T>& y) : Node<T>(x, y) {}

    /**
     * Compute gradients of z = x + y.
     * Local derivatives are 1, so both inputs receive `propagated_grad`.
     * Saved tensors are unused.
     *
     * @param propagated_grad Upstream gradient dL/dz.
     * @return `{dL/dx, dL/dy}`.
     */
    std::vector<Tensor<T>> apply(const Tensor<T>& propagated_grad) override {
        return {propagated_grad, propagated_grad};
    }
};

template <typename T>
class MultiplyBackward : public Node<T> {
public:
    /**
     * Construct the backward node for element-wise multiplication.
     * Aliases `x` and `y` onto `saved_tensors`.
     *
     * @param x Left forward operand.
     * @param y Right forward operand.
     */
    MultiplyBackward(const Tensor<T>& x, const Tensor<T>& y) : Node<T>(x, y) {}

    /**
     * Compute gradients of z = x * y.
     * Uses saved `x` and `y`: dL/dx = y * grad, dL/dy = x * grad.
     *
     * @param propagated_grad Upstream gradient dL/dz.
     * @return `{dL/dx, dL/dy}`.
     *
     * @throws std::invalid_argument on a shape mismatch with `propagated_grad`.
     */
    std::vector<Tensor<T>> apply(const Tensor<T>& propagated_grad) override {
        const Tensor<T>& x = this->saved_tensors[0];
        const Tensor<T>& y = this->saved_tensors[1];
        return {
            propagated_grad.multiply(y),
            x.multiply(propagated_grad)
        };
    }
};

template <typename T>
class NegationBackward : public Node<T> {
public:
    /**
     * Construct the backward node for negation.
     * Aliases `x` onto `saved_tensors`.
     *
     * @param x Forward operand of `-x`.
     */
    explicit NegationBackward(const Tensor<T>& x) : Node<T>(x) {}

    /**
     * Compute the gradient of z = -x.
     * Returns `-propagated_grad`. Saved `x` is unused.
     *
     * @param propagated_grad Upstream gradient dL/dz.
     * @return `{dL/dx}`.
     */
    std::vector<Tensor<T>> apply(const Tensor<T>& propagated_grad) override {
        return {propagated_grad.neg()};
    }
};

template <typename T>
class SubtractBackward : public Node<T> {
public:
    /**
     * Construct the backward node for element-wise subtraction.
     * Aliases `x` and `y` onto `saved_tensors`.
     *
     * @param x Left forward operand.
     * @param y Right forward operand.
     */
    SubtractBackward(const Tensor<T>& x, const Tensor<T>& y) : Node<T>(x, y) {}

    /**
     * Compute gradients of z = x - y.
     * Returns `{grad, -grad}`. Saved tensors are unused.
     *
     * @param propagated_grad Upstream gradient dL/dz.
     * @return `{dL/dx, dL/dy}`.
     */
    std::vector<Tensor<T>> apply(const Tensor<T>& propagated_grad) override {
        return {propagated_grad, propagated_grad.neg()};
    }
};

template <typename T>
class DivideBackward : public Node<T> {
public:
    /**
     * Construct the backward node for element-wise division.
     * Aliases `x` and `y` onto `saved_tensors`.
     *
     * @param x Numerator of the forward divide.
     * @param y Denominator of the forward divide.
     */
    DivideBackward(const Tensor<T>& x, const Tensor<T>& y) : Node<T>(x, y) {}

    /**
     * Compute gradients of z = x / y.
     * Uses saved `x` and `y`: dL/dx = grad / y, dL/dy = -x / y^2 * grad.
     *
     * @param propagated_grad Upstream gradient dL/dz.
     * @return `{dL/dx, dL/dy}`.
     *
     * @throws std::invalid_argument on a shape mismatch between operands.
     * @throws std::runtime_error on division by zero.
     */
    std::vector<Tensor<T>> apply(const Tensor<T>& propagated_grad) override {
        const Tensor<T>& x = this->saved_tensors[0];
        const Tensor<T>& y = this->saved_tensors[1];
        const Tensor<T> grad_x = propagated_grad.divide(y);
        const Tensor<T> grad_y = x.divide(y.power(static_cast<T>(2))).neg().multiply(propagated_grad);
        return {grad_x, grad_y};
    }
};

template <typename T>
class PowerBackward : public Node<T> {
    T exponent_;

public:
    /**
     * Construct the backward node for element-wise power with a scalar exponent.
     * Aliases `x` and stores `exponent`.
     *
     * @param x Forward base tensor.
     * @param exponent Scalar exponent from the forward pass.
     */
    PowerBackward(const Tensor<T>& x, const T exponent) : Node<T>(x), exponent_(exponent) {}

    /**
     * Compute the gradient of z = x^c for stored scalar `exponent_` (c).
     * Uses saved `x`: dL/dx = c * x^(c-1) * grad.
     *
     * @param propagated_grad Upstream gradient dL/dz.
     * @return `{dL/dx}`.
     *
     * @throws std::invalid_argument on a shape mismatch with `propagated_grad`.
     */
    std::vector<Tensor<T>> apply(const Tensor<T>& propagated_grad) override {
        const Tensor<T>& x = this->saved_tensors[0];
        const Tensor<T> coeff(x.shape(), exponent_, false);
        const Tensor<T> grad_x = x.power(exponent_ - static_cast<T>(1)).multiply(coeff).multiply(propagated_grad);
        return {grad_x};
    }
};

template <typename T>
class AbsBackward : public Node<T> {
public:
    /**
     * Construct the backward node for absolute value.
     * Aliases `x` onto `saved_tensors`.
     *
     * @param x Forward operand of abs.
     */
    explicit AbsBackward(const Tensor<T>& x) : Node<T>(x) {}

    /**
     * Compute the gradient of z = abs(x).
     * Uses saved `x` to form sign(x) (1, 0, or -1) and multiplies by `propagated_grad`.
     *
     * @param propagated_grad Upstream gradient dL/dz.
     * @return `{dL/dx}`.
     *
     * @throws std::invalid_argument on a shape mismatch with `propagated_grad`.
     */
    std::vector<Tensor<T>> apply(const Tensor<T>& propagated_grad) override {
        const Tensor<T>& x = this->saved_tensors[0];
        std::vector<T> sign_storage(static_cast<size_t>(x.numel()));
        for (size_t index = 0; index < sign_storage.size(); ++index) {
            const T value = x.data()[index];
            if (value > static_cast<T>(0))
                sign_storage[index] = static_cast<T>(1);
            else if (value < static_cast<T>(0))
                sign_storage[index] = static_cast<T>(-1);
            else
                sign_storage[index] = static_cast<T>(0);
        }
        const Tensor<T> sign_x = Tensor<T>::from_operation_result(
            x.shape(), std::move(sign_storage), false, nullptr
        );
        const Tensor<T> grad_x = sign_x.multiply(propagated_grad);
        return {grad_x};
    }
};

template <typename T>
class ExpBackward : public Node<T> {
public:
    /**
     * Construct the backward node for exp.
     * Aliases `x` onto `saved_tensors`.
     *
     * @param x Forward operand of exp.
     */
    explicit ExpBackward(const Tensor<T>& x) : Node<T>(x) {}

    /**
     * Compute the gradient of z = exp(x).
     * Uses saved `x`: dL/dx = exp(x) * grad.
     *
     * @param propagated_grad Upstream gradient dL/dz.
     * @return `{dL/dx}`.
     *
     * @throws std::invalid_argument on a shape mismatch with `propagated_grad`.
     */
    std::vector<Tensor<T>> apply(const Tensor<T>& propagated_grad) override {
        const Tensor<T>& x = this->saved_tensors[0];
        const Tensor<T> grad_x = x.exp().multiply(propagated_grad);
        return {grad_x};
    }
};

template <typename T>
class SqrtBackward : public Node<T> {
public:
    /**
     * Construct the backward node for square root.
     * Aliases `x` onto `saved_tensors`.
     *
     * @param x Forward operand of sqrt.
     */
    explicit SqrtBackward(const Tensor<T>& x) : Node<T>(x) {}

    /**
     * Compute the gradient of z = sqrt(x).
     * Uses saved `x` and `propagated_grad` storage: dL/dx = 0.5 / sqrt(x) * grad.
     *
     * @param propagated_grad Upstream gradient dL/dz.
     * @return `{dL/dx}`.
     */
    std::vector<Tensor<T>> apply(const Tensor<T>& propagated_grad) override {
        const Tensor<T>& x = this->saved_tensors[0];
        std::vector<T> storage(static_cast<size_t>(x.numel()));
        const auto& xd = x.data();
        const auto& gd = propagated_grad.data();
        for (size_t i = 0; i < storage.size(); ++i) {
            const T s = static_cast<T>(std::sqrt(static_cast<double>(xd[i])));
            storage[i] = gd[i] * static_cast<T>(0.5) / s;
        }
        return {Tensor<T>::from_operation_result(x.shape(), std::move(storage), false, nullptr)};
    }
};

template <typename T>
class LogBackward : public Node<T> {
public:
    /**
     * Construct the backward node for natural log.
     * Aliases `x` onto `saved_tensors`.
     *
     * @param x Forward operand of log.
     */
    explicit LogBackward(const Tensor<T>& x) : Node<T>(x) {}

    /**
     * Compute the gradient of z = log(x).
     * Uses saved `x`: dL/dx = grad / x.
     *
     * @param propagated_grad Upstream gradient dL/dz.
     * @return `{dL/dx}`.
     *
     * @throws std::invalid_argument on a shape mismatch with `propagated_grad`.
     * @throws std::runtime_error on division by zero.
     */
    std::vector<Tensor<T>> apply(const Tensor<T>& propagated_grad) override {
        const Tensor<T>& x = this->saved_tensors[0];
        const Tensor<T> grad_x = propagated_grad.divide(x);
        return {grad_x};
    }
};

template <typename T>
class SinBackward : public Node<T> {
public:
    /**
     * Construct the backward node for sine.
     * Aliases `x` onto `saved_tensors`.
     *
     * @param x Forward operand of sin.
     */
    explicit SinBackward(const Tensor<T>& x) : Node<T>(x) {}

    /**
     * Compute the gradient of z = sin(x).
     * Uses saved `x`: dL/dx = cos(x) * grad.
     *
     * @param propagated_grad Upstream gradient dL/dz.
     * @return `{dL/dx}`.
     *
     * @throws std::invalid_argument on a shape mismatch with `propagated_grad`.
     */
    std::vector<Tensor<T>> apply(const Tensor<T>& propagated_grad) override {
        const Tensor<T>& x = this->saved_tensors[0];
        const Tensor<T> grad_x = x.cos().multiply(propagated_grad);
        return {grad_x};
    }
};

template <typename T>
class CosBackward : public Node<T> {
public:
    /**
     * Construct the backward node for cosine.
     * Aliases `x` onto `saved_tensors`.
     *
     * @param x Forward operand of cos.
     */
    explicit CosBackward(const Tensor<T>& x) : Node<T>(x) {}

    /**
     * Compute the gradient of z = cos(x).
     * Uses saved `x`: dL/dx = -sin(x) * grad.
     *
     * @param propagated_grad Upstream gradient dL/dz.
     * @return `{dL/dx}`.
     *
     * @throws std::invalid_argument on a shape mismatch with `propagated_grad`.
     */
    std::vector<Tensor<T>> apply(const Tensor<T>& propagated_grad) override {
        const Tensor<T>& x = this->saved_tensors[0];
        const Tensor<T> grad_x = x.sin().neg().multiply(propagated_grad);
        return {grad_x};
    }
};

template <typename T>
class TanBackward : public Node<T> {
public:
    /**
     * Construct the backward node for tangent.
     * Aliases `x` onto `saved_tensors`.
     *
     * @param x Forward operand of tan.
     */
    explicit TanBackward(const Tensor<T>& x) : Node<T>(x) {}

    /**
     * Compute the gradient of z = tan(x).
     * Uses saved `x`: dL/dx = grad / cos(x)^2.
     *
     * @param propagated_grad Upstream gradient dL/dz.
     * @return `{dL/dx}`.
     *
     * @throws std::invalid_argument on a shape mismatch with `propagated_grad`.
     * @throws std::runtime_error on division by zero.
     */
    std::vector<Tensor<T>> apply(const Tensor<T>& propagated_grad) override {
        const Tensor<T>& x = this->saved_tensors[0];
        const Tensor<T> grad_x = propagated_grad.divide(x.cos().power(static_cast<T>(2)));
        return {grad_x};
    }
};

template <typename T>
class SinhBackward : public Node<T> {
public:
    /**
     * Construct the backward node for hyperbolic sine.
     * Aliases `x` onto `saved_tensors`.
     *
     * @param x Forward operand of sinh.
     */
    explicit SinhBackward(const Tensor<T>& x) : Node<T>(x) {}

    /**
     * Compute the gradient of z = sinh(x).
     * Uses saved `x`: dL/dx = cosh(x) * grad.
     *
     * @param propagated_grad Upstream gradient dL/dz.
     * @return `{dL/dx}`.
     *
     * @throws std::invalid_argument on a shape mismatch with `propagated_grad`.
     */
    std::vector<Tensor<T>> apply(const Tensor<T>& propagated_grad) override {
        const Tensor<T>& x = this->saved_tensors[0];
        const Tensor<T> grad_x = x.cosh().multiply(propagated_grad);
        return {grad_x};
    }
};

template <typename T>
class CoshBackward : public Node<T> {
public:
    /**
     * Construct the backward node for hyperbolic cosine.
     * Aliases `x` onto `saved_tensors`.
     *
     * @param x Forward operand of cosh.
     */
    explicit CoshBackward(const Tensor<T>& x) : Node<T>(x) {}

    /**
     * Compute the gradient of z = cosh(x).
     * Uses saved `x`: dL/dx = sinh(x) * grad.
     *
     * @param propagated_grad Upstream gradient dL/dz.
     * @return `{dL/dx}`.
     *
     * @throws std::invalid_argument on a shape mismatch with `propagated_grad`.
     */
    std::vector<Tensor<T>> apply(const Tensor<T>& propagated_grad) override {
        const Tensor<T>& x = this->saved_tensors[0];
        const Tensor<T> grad_x = x.sinh().multiply(propagated_grad);
        return {grad_x};
    }
};

template <typename T>
class TanhBackward : public Node<T> {
public:
    /**
     * Construct the backward node for hyperbolic tangent.
     * Aliases `x` onto `saved_tensors`.
     *
     * @param x Forward operand of tanh.
     */
    explicit TanhBackward(const Tensor<T>& x) : Node<T>(x) {}

    /**
     * Compute the gradient of z = tanh(x).
     * Uses saved `x`: dL/dx = (1 - tanh(x)^2) * grad.
     *
     * @param propagated_grad Upstream gradient dL/dz.
     * @return `{dL/dx}`.
     *
     * @throws std::invalid_argument on a shape mismatch with `propagated_grad`.
     */
    std::vector<Tensor<T>> apply(const Tensor<T>& propagated_grad) override {
        const Tensor<T>& x = this->saved_tensors[0];
        const Tensor<T> ones(x.shape(), static_cast<T>(1), false);
        const Tensor<T> grad_x = ones.subtract(x.tanh().power(static_cast<T>(2))).multiply(propagated_grad);
        return {grad_x};
    }
};

template <typename T>
class SigmoidBackward : public Node<T> {
public:
    /**
     * Construct the backward node for sigmoid.
     * Aliases `x` onto `saved_tensors`.
     *
     * @param x Forward operand of sigmoid.
     */
    explicit SigmoidBackward(const Tensor<T>& x) : Node<T>(x) {}

    /**
     * Compute the gradient of z = sigmoid(x).
     * Uses saved `x`: dL/dx = sigmoid(x) * (1 - sigmoid(x)) * grad.
     *
     * @param propagated_grad Upstream gradient dL/dz.
     * @return `{dL/dx}`.
     *
     * @throws std::invalid_argument on a shape mismatch with `propagated_grad`.
     */
    std::vector<Tensor<T>> apply(const Tensor<T>& propagated_grad) override {
        const Tensor<T>& x = this->saved_tensors[0];
        const Tensor<T> ones(x.shape(), static_cast<T>(1), false);
        const Tensor<T> sigmoid_x = x.sigmoid();
        return {sigmoid_x.multiply(ones.subtract(sigmoid_x)).multiply(propagated_grad)};
    }
};

template <typename T>
class ReluBackward : public Node<T> {
public:
    /**
     * Construct the backward node for ReLU.
     * Aliases `x` onto `saved_tensors`.
     *
     * @param x Forward operand of relu.
     */
    explicit ReluBackward(const Tensor<T>& x) : Node<T>(x) {}

    /**
     * Compute the gradient of z = max(0, x).
     * Uses saved `x` to build a 0/1 mask (1 iff x > 0) and multiplies by `propagated_grad`.
     *
     * @param propagated_grad Upstream gradient dL/dz.
     * @return `{dL/dx}`.
     *
     * @throws std::invalid_argument on a shape mismatch with `propagated_grad`.
     */
    std::vector<Tensor<T>> apply(const Tensor<T>& propagated_grad) override {
        const Tensor<T>& x = this->saved_tensors[0];
        std::vector<T> mask(static_cast<size_t>(x.numel()));
        for (size_t i = 0; i < mask.size(); ++i)
            mask[i] = x.data()[i] > static_cast<T>(0) ? static_cast<T>(1) : static_cast<T>(0);
        const Tensor<T> mask_tensor = Tensor<T>::from_operation_result(
            x.shape(), std::move(mask), false, nullptr);
        return {mask_tensor.multiply(propagated_grad)};
    }
};

template <typename T>
class SiLUBackward : public Node<T> {
public:
    /**
     * Construct the backward node for SiLU (x * sigmoid(x)).
     * Aliases `x` onto `saved_tensors`.
     *
     * @param x Forward operand of SiLU.
     */
    explicit SiLUBackward(const Tensor<T>& x) : Node<T>(x) {}

    /**
     * Compute the gradient of z = x * sigmoid(x).
     * Uses saved `x`: dL/dx = sigmoid(x) * (1 + x * (1 - sigmoid(x))) * grad.
     *
     * @param propagated_grad Upstream gradient dL/dz.
     * @return `{dL/dx}`.
     *
     * @throws std::invalid_argument on a shape mismatch with `propagated_grad`.
     */
    std::vector<Tensor<T>> apply(const Tensor<T>& propagated_grad) override {
        const Tensor<T>& x = this->saved_tensors[0];
        const Tensor<T> ones(x.shape(), static_cast<T>(1), false);
        const Tensor<T> sigmoid_x = x.sigmoid();
        const Tensor<T> local_grad = sigmoid_x.multiply(ones.add(x.multiply(ones.subtract(sigmoid_x))));
        return {local_grad.multiply(propagated_grad)};
    }
};

template <typename T>
class GELUBackward : public Node<T> {
public:
    /**
     * Construct the backward node for tanh-approximation GELU.
     * Aliases `x` onto `saved_tensors`.
     *
     * @param x Forward operand of GELU.
     */
    explicit GELUBackward(const Tensor<T>& x) : Node<T>(x) {}

    /**
     * Compute the gradient of tanh GELU.
     * Uses saved `x` with u = k*(x + c*x^3), k = sqrt(2/pi), c = 0.044715:
     * local grad is 0.5*(1 + tanh(u)) + 0.5*x * sech^2(u) * k*(1 + 3*c*x^2).
     *
     * @param propagated_grad Upstream gradient dL/dz.
     * @return `{dL/dx}`.
     *
     * @throws std::invalid_argument on a shape mismatch with `propagated_grad`.
     */
    std::vector<Tensor<T>> apply(const Tensor<T>& propagated_grad) override {
        const Tensor<T>& x = this->saved_tensors[0];
        const T k         = static_cast<T>(std::sqrt(2.0 / M_PI));
        constexpr T c     = static_cast<T>(0.044715);
        constexpr T half  = static_cast<T>(0.5);
        constexpr T one   = static_cast<T>(1);
        constexpr T three = static_cast<T>(3);

        const size_t n = static_cast<size_t>(x.numel());
        std::vector<T> grad_storage(n);
        for (size_t i = 0; i < n; ++i) {
            const T xi     = x.data()[i];
            const T u      = k * (xi + c * static_cast<T>(std::pow(xi, 3)));
            const T tanh_u = static_cast<T>(std::tanh(u));
            const T sech2  = one - tanh_u * tanh_u;
            const T du_dx  = k * (one + three * c * xi * xi);
            grad_storage[i] = half * (one + tanh_u) + half * xi * sech2 * du_dx;
        }
        const Tensor<T> local_grad = Tensor<T>::from_operation_result(
            x.shape(), std::move(grad_storage), false, nullptr);
        return {local_grad.multiply(propagated_grad)};
    }
};

} // namespace autograd
