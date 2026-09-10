#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <random>
#include <stdexcept>
#include <vector>

#include "../tensor/tensor.h"
#include "../autograd/grad_context.h"
#include "../autograd/backward_ops/backward_dropout_ops.h"

namespace ops {

/**
 * Apply inverted dropout. If `training` is false or `p` is 0, returns `x`
 * unchanged; otherwise multiplies packed elements by 0 or `1/(1-p)` and
 * attaches `DropoutBackward` when grad is enabled. `seed` is used only in
 * training mode so tests can be deterministic.
 *
 * @param x The tensor to drop out.
 * @param p Drop probability in `[0, 1)`.
 * @param training Whether dropout is active.
 * @param seed Optional RNG seed used only when training and `p > 0`.
 * @return `x` unchanged, or a new tensor of the same shape with dropped values.
 *
 * @throws std::invalid_argument if `p` is outside `[0, 1)`.
 */
template <typename T>
tensor::Tensor<T> dropout(
    const tensor::Tensor<T>& x,
    double p,
    bool training,
    std::optional<uint64_t> seed = std::nullopt) {
    if (p < 0.0 || p >= 1.0)
        throw std::invalid_argument("dropout: p must be in [0, 1)");
    if (!training || p == 0.0)
        return x;

    const auto xc = x.contiguous();
    const T scale = static_cast<T>(1.0 / (1.0 - p));
    std::vector<T> storage(static_cast<size_t>(xc.numel()));
    std::vector<T> mask_storage(static_cast<size_t>(xc.numel()));
    std::mt19937_64 rng{seed.has_value() ? *seed : std::random_device{}()};
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    const auto& xd = xc.data();
    for (size_t i = 0; i < storage.size(); ++i) {
        const T m = dist(rng) < p ? T{0} : scale;
        mask_storage[i] = m;
        storage[i] = xd[i] * m;
    }

    const bool requires_grad = autograd::is_grad_enabled() && xc.requires_grad();
    tensor::Tensor<T> mask = tensor::Tensor<T>::from_operation_result(
        xc.shape(), std::move(mask_storage), false, nullptr);
    std::shared_ptr<autograd::Node<T>> grad_fn = nullptr;
    if (requires_grad)
        grad_fn = std::make_shared<autograd::DropoutBackward<T>>(xc, mask);

    return tensor::Tensor<T>::from_operation_result(
        xc.shape(), std::move(storage), requires_grad, std::move(grad_fn));
}

} // namespace ops
