#pragma once

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

} // namespace autograd