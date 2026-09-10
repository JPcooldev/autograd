#pragma once

#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

#include "../node.h"
#include "../../ops/conv_kernels.h"

namespace autograd {

using tensor::Tensor;

template <typename T>
class MaxPoolBackward : public Node<T> {
    std::vector<int64_t> argmax_;
    std::vector<int64_t> in_shape_;

public:
    /**
     * Construct the backward node for max pooling.
     * Aliases `input` for the graph edge and stores flattened argmax indices
     * plus the input shape.
     *
     * @param input Forward pooled input (edge only; unused in apply).
     * @param argmax Flat input index of the max for each output element.
     */
    MaxPoolBackward(const Tensor<T>& input, std::vector<int64_t> argmax)
        : Node<T>(input), argmax_(std::move(argmax)), in_shape_(input.shape()) {}

    /**
     * Scatter `propagated_grad` onto the max locations in the input.
     * Uses stored `argmax_` and `in_shape_`; saved input values are unused.
     *
     * @param propagated_grad Upstream gradient dL/d(pool output).
     * @return `{dL/d(input)}` with zeros except at argmax positions.
     */
    std::vector<Tensor<T>> apply(const Tensor<T>& propagated_grad) override {
        std::vector<T> g_in(static_cast<size_t>(Tensor<T>::compute_numel(in_shape_)), T{0});
        const auto g = propagated_grad.contiguous();
        const auto& gd = g.data();
        for (int64_t i = 0; i < g.numel(); ++i)
            g_in[static_cast<size_t>(argmax_[static_cast<size_t>(i)])] += gd[static_cast<size_t>(i)];
        return {Tensor<T>::from_operation_result(in_shape_, std::move(g_in), false, nullptr)};
    }
};

template <typename T>
class AvgPoolBackward : public Node<T> {
    int64_t kernel_;
    int64_t stride_;
    int64_t padding_;
    std::vector<int64_t> in_shape_;
    std::vector<int64_t> out_shape_;

public:
    /**
     * Construct the backward node for average pooling.
     * Aliases `input` for the graph edge and stores kernel, stride, padding,
     * and input/output shapes.
     *
     * @param input Forward pooled input (edge only; unused in apply).
     * @param kernel Spatial kernel size (same on every pooled axis).
     * @param stride Pool stride.
     * @param padding Spatial padding.
     * @param out_shape Forward output shape.
     */
    AvgPoolBackward(
        const Tensor<T>& input,
        int64_t kernel, int64_t stride, int64_t padding,
        std::vector<int64_t> out_shape
    )
        : Node<T>(input), kernel_(kernel), stride_(stride), padding_(padding),
          in_shape_(input.shape()), out_shape_(std::move(out_shape)) {}

    /**
     * Distribute each output gradient uniformly over the pooling window.
     * Uses stored kernel/stride/padding and shapes; saved input values unused.
     * Positions that fell in padding are skipped.
     *
     * @param propagated_grad Upstream gradient dL/d(pool output).
     * @return `{dL/d(input)}`.
     */
    std::vector<Tensor<T>> apply(const Tensor<T>& propagated_grad) override {
        using namespace ops::conv_detail;
        const int64_t D = static_cast<int64_t>(in_shape_.size()) - 2;
        std::vector<T> g_in(static_cast<size_t>(Tensor<T>::compute_numel(in_shape_)), T{0});
        const auto g = propagated_grad.contiguous();
        const auto& gd = g.data();
        const auto in_st = contig_strides(in_shape_);
        const auto out_st = contig_strides(out_shape_);
        const int64_t kvol = [&] {
            int64_t p = 1;
            for (int64_t d = 0; d < D; ++d)
                p *= kernel_;
            return p;
        }();
        const T inv = static_cast<T>(1) / static_cast<T>(kvol);
        const int64_t out_n = product(out_shape_, 0, static_cast<int64_t>(out_shape_.size()));
        std::vector<int64_t> oidx;
        for (int64_t f = 0; f < out_n; ++f) {
            unravel(f, out_shape_, oidx);
            const int64_t n = oidx[0];
            const int64_t c = oidx[1];
            const T go = gd[static_cast<size_t>(f)] * inv;
            const int64_t kmax = kvol;
            for (int64_t kf = 0; kf < kmax; ++kf) {
                int64_t remaining = kf;
                bool inside = true;
                int64_t in_off = n * in_st[0] + c * in_st[1];
                for (int64_t d = D - 1; d >= 0; --d) {
                    const int64_t kd = remaining % kernel_;
                    remaining /= kernel_;
                    const int64_t in_d =
                        oidx[static_cast<size_t>(2 + d)] * stride_ - padding_ + kd;
                    if (in_d < 0 || in_d >= in_shape_[static_cast<size_t>(2 + d)]) {
                        inside = false;
                        break;
                    }
                    in_off += in_d * in_st[static_cast<size_t>(2 + d)];
                }
                if (inside)
                    g_in[static_cast<size_t>(in_off)] += go;
            }
        }
        return {Tensor<T>::from_operation_result(in_shape_, std::move(g_in), false, nullptr)};
    }
};

} // namespace autograd
