#pragma once

#include "optimizer.h"

namespace nn {
namespace optim {

template <typename T>
class SGD : public Optimizer<T> {
public:
    SGD(
        std::vector<tensor::Tensor<T>*> parameters,
        double learning_rate,
        double weight_decay = 0.0
    ) : Optimizer<T>(parameters, learning_rate, weight_decay)
    {}

    ~SGD() = default;

    void step() override
    {
        const double learning_rate = this->learning_rate_;
        const double weight_decay = this->weight_decay_;

        for (auto* param : this->parameters_) 
        {
            if (!param->requires_grad())
                continue;

            const tensor::Tensor<T>* grad_tensor = param->grad();
            if (!grad_tensor)
                continue;

            auto& data = param->data();
            const auto& grad = grad_tensor->data();
            const size_t n = data.size();

            for (size_t i = 0; i < n; ++i)
                // theta_t = theta_{t-1} - lr * (g_t + wd * theta_{t-1})
                data[i] -= static_cast<T>(learning_rate * (
                    static_cast<double>(grad[i]) + weight_decay * static_cast<double>(data[i])));
        }
    }
};

} // namespace optim
} // namespace nn
