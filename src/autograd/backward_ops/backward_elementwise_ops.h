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
    AddBackward(const Tensor<T>& x, const Tensor<T>& y) : Node<T>(x, y) {}

    /*
    z = x + y
    dz/dx = 1 * grad_z
    dz/dy = 1 * grad_z
    */
    std::vector<Tensor<T>> apply(const Tensor<T>& propagated_grad) override {
        return {propagated_grad, propagated_grad};
    }
};

template <typename T>
class MultiplyBackward : public Node<T> {
public:
    MultiplyBackward(const Tensor<T>& x, const Tensor<T>& y) : Node<T>(x, y) {}

    /*
    z = x * y
    dz/dx = y * grad_z
    dz/dy = x * grad_z
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
    explicit NegationBackward(const Tensor<T>& x) : Node<T>(x) {}

    /*
    z = -x
    dz/dx = -1 * grad_z
    */
    std::vector<Tensor<T>> apply(const Tensor<T>& propagated_grad) override {
        return {propagated_grad.neg()};
    }
};

template <typename T>
class SubtractBackward : public Node<T> {
public:
    SubtractBackward(const Tensor<T>& x, const Tensor<T>& y) : Node<T>(x, y) {}

    /*
    z = x - y
    dz/dx = 1 * grad_z
    dz/dy = -1 * grad_z
    */
    std::vector<Tensor<T>> apply(const Tensor<T>& propagated_grad) override {
        return {propagated_grad, propagated_grad.neg()};
    }
};

template <typename T>
class DivideBackward : public Node<T> {
public:
    DivideBackward(const Tensor<T>& x, const Tensor<T>& y) : Node<T>(x, y) {}

    /*
    z = x / y
    dz/dx = (1 / y) * grad_z
    dz/dy = (-x / y^2) * grad_z
    */
    std::vector<Tensor<T>> apply(const Tensor<T>& propagated_grad) override {
        const Tensor<T>& x = this->saved_tensors[0];
        const Tensor<T>& y = this->saved_tensors[1];
        // dz/dx = (1 / y) * grad_z
        const Tensor<T> grad_x = propagated_grad.divide(y);
        // dz/dy = (-x / y^2) * grad_z
        const Tensor<T> grad_y = x.divide(y.power(static_cast<T>(2))).neg().multiply(propagated_grad);
        return {grad_x, grad_y};
    }
};

template <typename T>
class PowerBackward : public Node<T> {
public:
    PowerBackward(const Tensor<T>& x, const T exponent) : Node<T>(x), exponent_(exponent) {}

    /*
    z = x ^ constant
    dz/dx = constant * x ^ (constant - 1) * grad_z
    */
    std::vector<Tensor<T>> apply(const Tensor<T>& propagated_grad) override {
        const Tensor<T>& x = this->saved_tensors[0];
        // dz/dx = constant * x ^ (constant - 1) * grad_z
        const Tensor<T> coeff(x.shape(), exponent_, false);
        const Tensor<T> grad_x = x.power(exponent_ - static_cast<T>(1)).multiply(coeff).multiply(propagated_grad);
        return {grad_x};
    }

private:
    T exponent_;
};

template <typename T>
class AbsBackward : public Node<T> {
public:
    explicit AbsBackward(const Tensor<T>& x) : Node<T>(x) {}

    /*
    z = abs(x)
    dz/dx = sign(x) * grad_z, 
    where sign(x) is 1 if x > 0, 
                     0 if x == 0, 
                    -1 if x < 0
    */
    std::vector<Tensor<T>> apply(const Tensor<T>& propagated_grad) override {
        const Tensor<T>& x = this->saved_tensors[0];
        std::vector<T> sign_storage(static_cast<size_t>(x.numel()));
        for (size_t index = 0; index < sign_storage.size(); ++index) {
            const T value = x.data()[index];
            if (value > static_cast<T>(0)) {
                sign_storage[index] = static_cast<T>(1);
            } else if (value < static_cast<T>(0)) {
                sign_storage[index] = static_cast<T>(-1);
            } else {
                sign_storage[index] = static_cast<T>(0);
            }
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
    explicit ExpBackward(const Tensor<T>& x) : Node<T>(x) {}

    /*
    z = exp(x)
    dz/dx = exp(x) * grad_z
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
    explicit SqrtBackward(const Tensor<T>& x) : Node<T>(x) {}

    /*
    z = sqrt(x)
    dz/dx = 0.5 / sqrt(x) * grad_z
    */

    std::vector<Tensor<T>> apply(const Tensor<T>& propagated_grad) override {
        const Tensor<T>& x = this->saved_tensors[0];
        std::vector<T> storage(static_cast<size_t>(x.numel()));
        const auto& xd = x.data();
        const auto& gd = propagated_grad.data();
        for (size_t i = 0; i < storage.size(); ++i) {
            // sqrt(x)
            const T s = static_cast<T>(std::sqrt(static_cast<double>(xd[i])));
            // dz/dx = 0.5 / sqrt(x) * grad_z
            storage[i] = gd[i] * static_cast<T>(0.5) / s;
        }
        return {Tensor<T>::from_operation_result(x.shape(), std::move(storage), false, nullptr)};
    }
};

template <typename T>
class LogBackward : public Node<T> {
public:
    explicit LogBackward(const Tensor<T>& x) : Node<T>(x) {}

    /*
    z = log(x)
    dz/dx = 1 / x * grad_z
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
    explicit SinBackward(const Tensor<T>& x) : Node<T>(x) {}

    /*
    z = sin(x)
    dz/dx = cos(x) * grad_z
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
    explicit CosBackward(const Tensor<T>& x) : Node<T>(x) {}

    /*
    z = cos(x)
    dz/dx = -sin(x) * grad_z
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
    explicit TanBackward(const Tensor<T>& x) : Node<T>(x) {}

    /*
    z = tan(x)
    dz/dx = 1 / cos^2(x) * grad_z
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
    explicit SinhBackward(const Tensor<T>& x) : Node<T>(x) {}

    /*
    z = sinh(x)
    dz/dx = cosh(x) * grad_z
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
    explicit CoshBackward(const Tensor<T>& x) : Node<T>(x) {}

    /*
    z = cosh(x)
    dz/dx = sinh(x) * grad_z
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
    explicit TanhBackward(const Tensor<T>& x) : Node<T>(x) {}

    /*
    z = tanh(x)
    dz/dx = (1 - tanh(x)^2) * grad_z
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
    explicit SigmoidBackward(const Tensor<T>& x) : Node<T>(x) {}

    /*
    z = sigmoid(x)
    dz/dx = sigmoid(x) * (1 - sigmoid(x)) * grad_z
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
    explicit ReluBackward(const Tensor<T>& x) : Node<T>(x) {}

    /*
    z = relu(x) = max(0, x)
    dz/dx = 1 if x > 0, 
            0 if x <= 0
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
    explicit SiLUBackward(const Tensor<T>& x) : Node<T>(x) {}

    /*
    z = SiLU(x) = x * sigmoid(x)
    dz/dx = sigmoid(x) * (1 + x * (1 - sigmoid(x))) * grad_z
    */

    std::vector<Tensor<T>> apply(const Tensor<T>& propagated_grad) override {
        const Tensor<T>& x = this->saved_tensors[0];
        // d/dx SiLU(x) = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
        const Tensor<T> ones(x.shape(), static_cast<T>(1), false);
        const Tensor<T> sigmoid_x = x.sigmoid();
        const Tensor<T> local_grad = sigmoid_x.multiply(ones.add(x.multiply(ones.subtract(sigmoid_x))));
        return {local_grad.multiply(propagated_grad)};
    }
};

template <typename T>
class GELUBackward : public Node<T> {
public:
    explicit GELUBackward(const Tensor<T>& x) : Node<T>(x) {}

    /*
    z = GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    dz/dx = 0.5*(1 + tanh(u)) + 0.5*x * sech^2(u) * k*(1 + 3*c*x^2) * grad_z
    where u = k*(x + c*x^3), 
          k = sqrt(2/pi), 
          c = 0.044715
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