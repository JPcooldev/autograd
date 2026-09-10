#pragma once

#include <cmath>
#include <unordered_map>
#include <vector>

#include "optimizer.h"

namespace nn {
namespace optim {

template <typename T>
struct AdamWState {
    std::vector<T> m;
    std::vector<T> v;
    int64_t step = 0;
};

template <typename T>
class AdamW : public Optimizer<T> {
private:
    double beta1_;
    double beta2_;
    double eps_;
    std::unordered_map<tensor::Tensor<T>*, AdamWState<T>> state_;

public:
    /**
     * Construct AdamW with decoupled weight decay.
     * Forwards parameters to `Optimizer` (default λ = 0.01), stores β1, β2, and ε,
     * and allocates zero first- and second-moment buffers per parameter.
     *
     * @param parameters Non-empty list of parameter tensors to update.
     * @param learning_rate Step size η.
     * @param beta1 Exponential decay for the first moment. Default 0.9.
     * @param beta2 Exponential decay for the second moment. Default 0.999.
     * @param eps Denominator stabilizer ε. Default 1e-8.
     * @param weight_decay Decoupled decay coefficient λ applied as −η λ θ, not inside the moments. Default 0.01.
     *
     * @throws std::invalid_argument if `parameters` is empty.
     */
    AdamW(
        std::vector<tensor::Tensor<T>*> parameters,
        double learning_rate,
        double beta1 = 0.9,
        double beta2 = 0.999,
        double eps = 1e-8,
        double weight_decay = 0.01
    ) :
        Optimizer<T>(parameters, learning_rate, weight_decay),
        beta1_(beta1),
        beta2_(beta2),
        eps_(eps) {
        for (auto* param : this->parameters_) {
            const size_t n = param->data().size();
            state_[param] = { std::vector<T>(n, T{0}), std::vector<T>(n, T{0}) };
        }
    }

    /**
     * Destroy the AdamW optimizer.
     */
    ~AdamW() = default;

    /**
     * Apply one AdamW update with decoupled weight decay.
     * Moments use the raw gradient g (not g + λθ). After bias-corrected Adam,
     * θ ← θ − η m̂ / (√v̂ + ε) − η λ θ. Skips tensors without `requires_grad` or grad.
     */
    void step() override {
        const double learning_rate = this->learning_rate_;
        const double weight_decay = this->weight_decay_;
        const double beta1 = beta1_;
        const double beta2 = beta2_;
        const double eps = eps_;
        const double one_minus_beta1 = 1.0 - beta1;
        const double one_minus_beta2 = 1.0 - beta2;

        for (auto* param : this->parameters_) {
            if (!param->requires_grad())
                continue;

            const tensor::Tensor<T>* grad_tensor = param->grad();
            if (!grad_tensor)
                continue;

            auto& state = state_[param];
            auto& data  = param->data();
            const auto& grad = grad_tensor->data();
            const size_t n   = data.size();
            ++state.step;
            const double one_minus_beta1_pow_step = 1.0 - std::pow(beta1, static_cast<double>(state.step));
            const double one_minus_beta2_pow_step = 1.0 - std::pow(beta2, static_cast<double>(state.step));

            for (size_t i = 0; i < n; ++i) {
                const double g = static_cast<double>(grad[i]);

                // m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
                state.m[i] = static_cast<T>(beta1 * static_cast<double>(state.m[i]) + one_minus_beta1 * g);
                // v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
                state.v[i] = static_cast<T>(beta2 * static_cast<double>(state.v[i]) + one_minus_beta2 * g * g);

                // m_hat = m_t / (1 - beta1^t),  v_hat = v_t / (1 - beta2^t)
                const double m_hat = static_cast<double>(state.m[i]) / one_minus_beta1_pow_step;
                const double v_hat = static_cast<double>(state.v[i]) / one_minus_beta2_pow_step;
                // theta_t = theta_{t-1} - lr * m_hat / (sqrt(v_hat) + eps) - lr * wd * theta_{t-1}
                data[i] -= static_cast<T>(
                    learning_rate * m_hat / (std::sqrt(v_hat) + eps)
                    + learning_rate * weight_decay * static_cast<double>(data[i]));
            }
        }
    }
};

} // namespace optim
} // namespace nn
