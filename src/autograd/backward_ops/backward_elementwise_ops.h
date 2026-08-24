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

    std::vector<Tensor<T>> apply(const Tensor<T>& propagated_grad) override {
        return {propagated_grad, propagated_grad};
    }
};

template <typename T>
class MultiplyBackward : public Node<T> {
public:
    MultiplyBackward(const Tensor<T>& x, const Tensor<T>& y) : Node<T>(x, y) {}

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

    std::vector<Tensor<T>> apply(const Tensor<T>& propagated_grad) override {
        return {propagated_grad.neg()};
    }
};

template <typename T>
class SubtractBackward : public Node<T> {
public:
    SubtractBackward(const Tensor<T>& x, const Tensor<T>& y) : Node<T>(x, y) {}

    std::vector<Tensor<T>> apply(const Tensor<T>& propagated_grad) override {
        return {
            propagated_grad,
            propagated_grad.neg()
        };
    }
};

template <typename T>
class DivideBackward : public Node<T> {
public:
    DivideBackward(const Tensor<T>& x, const Tensor<T>& y) : Node<T>(x, y) {}

    std::vector<Tensor<T>> apply(const Tensor<T>& propagated_grad) override {
        const Tensor<T>& x = this->saved_tensors[0];
        const Tensor<T>& y = this->saved_tensors[1];
        const Tensor<T> grad_x = propagated_grad.divide(y);
        const Tensor<T> y_squared = y.power(static_cast<T>(2));
        const Tensor<T> x_over_y_squared = x.divide(y_squared);
        const Tensor<T> neg_x_over_y_squared = x_over_y_squared.neg();
        const Tensor<T> grad_y = neg_x_over_y_squared.multiply(propagated_grad);
        return {grad_x, grad_y};
    }
};

template <typename T>
class PowerBackward : public Node<T> {
public:
    PowerBackward(const Tensor<T>& x, const T exponent) : Node<T>(x), exponent_(exponent) {}

    std::vector<Tensor<T>> apply(const Tensor<T>& propagated_grad) override {
        const Tensor<T>& x = this->saved_tensors[0];
        const Tensor<T> exponent_tensor(x.shape(), exponent_, false);
        const Tensor<T> x_power = x.power(exponent_ - static_cast<T>(1));
        const Tensor<T> local_grad = exponent_tensor.multiply(x_power);
        const Tensor<T> grad_x = local_grad.multiply(propagated_grad);
        return {grad_x};
    }

private:
    T exponent_;
};

template <typename T>
class AbsBackward : public Node<T> {
public:
    explicit AbsBackward(const Tensor<T>& x) : Node<T>(x) {}

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
            x.shape(),
            std::move(sign_storage),
            false,
            nullptr
        );
        const Tensor<T> grad_x = sign_x.multiply(propagated_grad);
        return {grad_x};
    }
};

template <typename T>
class ExpBackward : public Node<T> {
public:
    explicit ExpBackward(const Tensor<T>& x) : Node<T>(x) {}

    std::vector<Tensor<T>> apply(const Tensor<T>& propagated_grad) override {
        const Tensor<T>& x = this->saved_tensors[0];
        const Tensor<T> exp_x = x.exp();
        const Tensor<T> grad_x = exp_x.multiply(propagated_grad);
        return {grad_x};
    }
};

template <typename T>
class SqrtBackward : public Node<T> {
public:
    explicit SqrtBackward(const Tensor<T>& x) : Node<T>(x) {}

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
    explicit LogBackward(const Tensor<T>& x) : Node<T>(x) {}

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

    std::vector<Tensor<T>> apply(const Tensor<T>& propagated_grad) override {
        const Tensor<T>& x = this->saved_tensors[0];
        const Tensor<T> cos_x = x.cos();
        const Tensor<T> grad_x = cos_x.multiply(propagated_grad);
        return {grad_x};
    }
};

template <typename T>
class CosBackward : public Node<T> {
public:
    explicit CosBackward(const Tensor<T>& x) : Node<T>(x) {}

    std::vector<Tensor<T>> apply(const Tensor<T>& propagated_grad) override {
        const Tensor<T>& x = this->saved_tensors[0];
        const Tensor<T> neg_sin_x = x.sin().neg();
        const Tensor<T> grad_x = neg_sin_x.multiply(propagated_grad);
        return {grad_x};
    }
};

template <typename T>
class TanBackward : public Node<T> {
public:
    explicit TanBackward(const Tensor<T>& x) : Node<T>(x) {}

    std::vector<Tensor<T>> apply(const Tensor<T>& propagated_grad) override {
        const Tensor<T>& x = this->saved_tensors[0];
        const Tensor<T> cos_x = x.cos();
        const Tensor<T> cos_squared = cos_x.power(static_cast<T>(2));
        const Tensor<T> ones(x.shape(), static_cast<T>(1), false);
        const Tensor<T> sec2_x = ones.divide(cos_squared);
        const Tensor<T> grad_x = sec2_x.multiply(propagated_grad);
        return {grad_x};
    }
};

template <typename T>
class SinhBackward : public Node<T> {
public:
    explicit SinhBackward(const Tensor<T>& x) : Node<T>(x) {}

    std::vector<Tensor<T>> apply(const Tensor<T>& propagated_grad) override {
        const Tensor<T>& x = this->saved_tensors[0];
        // d/dx sinh(x) = cosh(x)
        const Tensor<T> grad_x = x.cosh().multiply(propagated_grad);
        return {grad_x};
    }
};

template <typename T>
class CoshBackward : public Node<T> {
public:
    explicit CoshBackward(const Tensor<T>& x) : Node<T>(x) {}

    std::vector<Tensor<T>> apply(const Tensor<T>& propagated_grad) override {
        const Tensor<T>& x = this->saved_tensors[0];
        // d/dx cosh(x) = sinh(x)
        const Tensor<T> grad_x = x.sinh().multiply(propagated_grad);
        return {grad_x};
    }
};

template <typename T>
class TanhBackward : public Node<T> {
public:
    explicit TanhBackward(const Tensor<T>& x) : Node<T>(x) {}

    std::vector<Tensor<T>> apply(const Tensor<T>& propagated_grad) override {
        const Tensor<T>& x = this->saved_tensors[0];
        // d/dx tanh(x) = 1 - tanh(x)^2
        const Tensor<T> ones(x.shape(), static_cast<T>(1), false);
        const Tensor<T> tanh_x = x.tanh();
        const Tensor<T> local_grad = ones.subtract(tanh_x.power(static_cast<T>(2)));
        return {local_grad.multiply(propagated_grad)};
    }
};

template <typename T>
class SigmoidBackward : public Node<T> {
public:
    explicit SigmoidBackward(const Tensor<T>& x) : Node<T>(x) {}

    std::vector<Tensor<T>> apply(const Tensor<T>& propagated_grad) override {
        const Tensor<T>& x = this->saved_tensors[0];
        // d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))
        const Tensor<T> ones(x.shape(), static_cast<T>(1), false);
        const Tensor<T> sigmoid_x = ones.divide(ones.add(x.neg().exp()));
        const Tensor<T> local_grad = sigmoid_x.multiply(ones.subtract(sigmoid_x));
        return {local_grad.multiply(propagated_grad)};
    }
};

template <typename T>
class ReluBackward : public Node<T> {
public:
    explicit ReluBackward(const Tensor<T>& x) : Node<T>(x) {}

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

    std::vector<Tensor<T>> apply(const Tensor<T>& propagated_grad) override {
        const Tensor<T>& x = this->saved_tensors[0];
        // d/dx SiLU(x) = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
        const Tensor<T> ones(x.shape(), static_cast<T>(1), false);
        const Tensor<T> sigmoid_x = ones.divide(ones.add(x.neg().exp()));
        const Tensor<T> local_grad = sigmoid_x.multiply(ones.add(x.multiply(ones.subtract(sigmoid_x))));
        return {local_grad.multiply(propagated_grad)};
    }
};

template <typename T>
class GELUBackward : public Node<T> {
public:
    explicit GELUBackward(const Tensor<T>& x) : Node<T>(x) {}

    std::vector<Tensor<T>> apply(const Tensor<T>& propagated_grad) override {
        const Tensor<T>& x = this->saved_tensors[0];
        // GELU'(x) = 0.5*(1 + tanh(u)) + 0.5*x * sech^2(u) * k*(1 + 3*c*x^2)
        // where u = k*(x + c*x^3), k = sqrt(2/pi), c = 0.044715
        const T k         = static_cast<T>(std::sqrt(2.0 / M_PI));  // sqrt(2/pi)
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