#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "../node.h"
#include "../../ops/conv_kernels.h"

namespace autograd {

using tensor::Tensor;

template <typename T>
class ConvBackward : public Node<T> {
    int64_t stride_;
    int64_t padding_;
    int64_t dilation_;
    bool has_bias_;
    std::vector<int64_t> in_shape_;
    std::vector<int64_t> w_shape_;
    std::vector<int64_t> out_shape_;

public:
    /**
     * Construct the backward node for convolution.
     * Aliases `input` and `weight` (and `bias` when non-null) and stores
     * convolution hyperparameters plus input/weight/output shapes.
     *
     * @param input Forward input (N, Cin, *spatial).
     * @param weight Forward weight (Cout, Cin, *K).
     * @param bias Optional bias (Cout), or nullptr.
     * @param stride Convolution stride.
     * @param padding Spatial padding.
     * @param dilation Kernel dilation.
     * @param out_shape Forward output shape.
     */
    ConvBackward(
        const Tensor<T>& input,
        const Tensor<T>& weight,
        const Tensor<T>* bias,
        int64_t stride,
        int64_t padding,
        int64_t dilation,
        std::vector<int64_t> out_shape
    )
        : stride_(stride), padding_(padding), dilation_(dilation),
          has_bias_(bias != nullptr),
          in_shape_(input.shape()), w_shape_(weight.shape()),
          out_shape_(std::move(out_shape)) {
        const size_t n = has_bias_ ? 3 : 2;
        this->saved_tensors.reserve(n);
        this->next_edges.reserve(n);
        this->saved_tensors.emplace_back(Tensor<T>::alias(input));
        this->saved_tensors.emplace_back(Tensor<T>::alias(weight));
        this->next_edges.emplace_back(Node<T>::get_next_edge(input));
        this->next_edges.emplace_back(Node<T>::get_next_edge(weight));
        if (has_bias_) {
            this->saved_tensors.emplace_back(Tensor<T>::alias(*bias));
            this->next_edges.emplace_back(Node<T>::get_next_edge(*bias));
        }
    }

    /**
     * Compute convolution input, weight, and optional bias gradients.
     * Uses saved input and weight (contiguous copies) plus stored shapes and
     * `stride_` / `padding_` / `dilation_`; bias grad sums `propagated_grad`
     * over batch and spatial dims.
     *
     * @param propagated_grad Upstream gradient dL/d(output).
     * @return `{dL/d(input), dL/d(weight)}`, plus `dL/d(bias)` when a bias
     *         was saved.
     */
    std::vector<Tensor<T>> apply(const Tensor<T>& propagated_grad) override {
        using namespace ops::conv_detail;
        const Tensor<T>& input = this->saved_tensors[0];
        const Tensor<T>& weight = this->saved_tensors[1];
        const auto g = propagated_grad.contiguous();
        const auto in_c = input.contiguous();
        const auto w_c = weight.contiguous();

        std::vector<T> g_in(static_cast<size_t>(Tensor<T>::compute_numel(in_shape_)));
        std::vector<T> g_w(static_cast<size_t>(Tensor<T>::compute_numel(w_shape_)));
        conv_backward_input(g.data().data(), out_shape_,
                            w_c.data().data(), w_shape_,
                            g_in.data(), in_shape_,
                            stride_, padding_, dilation_);
        conv_backward_weight(in_c.data().data(), in_shape_,
                             g.data().data(), out_shape_,
                             g_w.data(), w_shape_,
                             stride_, padding_, dilation_);

        std::vector<Tensor<T>> grads;
        grads.push_back(Tensor<T>::from_operation_result(in_shape_, std::move(g_in), false, nullptr));
        grads.push_back(Tensor<T>::from_operation_result(w_shape_, std::move(g_w), false, nullptr));
        if (has_bias_) {
            std::vector<T> g_b(static_cast<size_t>(w_shape_[0]));
            conv_backward_bias(g.data().data(), out_shape_, g_b.data());
            grads.push_back(Tensor<T>::from_operation_result(
                {w_shape_[0]}, std::move(g_b), false, nullptr));
        }
        return grads;
    }
};

template <typename T>
class ConvTransposeBackward : public Node<T> {
    int64_t stride_;
    int64_t padding_;
    int64_t dilation_;
    bool has_bias_;
    std::vector<int64_t> in_shape_;
    std::vector<int64_t> w_shape_;
    std::vector<int64_t> out_shape_;

public:
    /**
     * Construct the backward node for transposed convolution.
     * Aliases `input` and `weight` (and `bias` when non-null) and stores
     * hyperparameters plus input/weight/output shapes.
     *
     * @param input Forward input (N, Cin, *spatial).
     * @param weight Forward weight (Cin, Cout, *K).
     * @param bias Optional bias (Cout), or nullptr.
     * @param stride Transposed-convolution stride.
     * @param padding Spatial padding.
     * @param dilation Kernel dilation.
     * @param out_shape Forward output shape.
     */
    ConvTransposeBackward(
        const Tensor<T>& input,
        const Tensor<T>& weight,
        const Tensor<T>* bias,
        int64_t stride,
        int64_t padding,
        int64_t dilation,
        std::vector<int64_t> out_shape
    )
        : stride_(stride), padding_(padding), dilation_(dilation),
          has_bias_(bias != nullptr),
          in_shape_(input.shape()), w_shape_(weight.shape()),
          out_shape_(std::move(out_shape)) {
        const size_t n = has_bias_ ? 3 : 2;
        this->saved_tensors.reserve(n);
        this->next_edges.reserve(n);
        this->saved_tensors.emplace_back(Tensor<T>::alias(input));
        this->saved_tensors.emplace_back(Tensor<T>::alias(weight));
        this->next_edges.emplace_back(Node<T>::get_next_edge(input));
        this->next_edges.emplace_back(Node<T>::get_next_edge(weight));
        if (has_bias_) {
            this->saved_tensors.emplace_back(Tensor<T>::alias(*bias));
            this->next_edges.emplace_back(Node<T>::get_next_edge(*bias));
        }
    }

    /**
     * Compute transposed-convolution input, weight, and optional bias gradients.
     * Uses saved input and weight plus stored shapes and hyperparameters;
     * bias length is `w_shape_[1]` (Cout).
     *
     * @param propagated_grad Upstream gradient dL/d(output).
     * @return `{dL/d(input), dL/d(weight)}`, plus `dL/d(bias)` when a bias
     *         was saved.
     */
    std::vector<Tensor<T>> apply(const Tensor<T>& propagated_grad) override {
        using namespace ops::conv_detail;
        const Tensor<T>& input = this->saved_tensors[0];
        const Tensor<T>& weight = this->saved_tensors[1];
        const auto g = propagated_grad.contiguous();
        const auto in_c = input.contiguous();
        const auto w_c = weight.contiguous();

        std::vector<T> g_in(static_cast<size_t>(Tensor<T>::compute_numel(in_shape_)));
        std::vector<T> g_w(static_cast<size_t>(Tensor<T>::compute_numel(w_shape_)));
        conv_transpose_backward_input(
            g.data().data(), out_shape_,
            w_c.data().data(), w_shape_,
            g_in.data(), in_shape_,
            stride_, padding_, dilation_);
        conv_transpose_backward_weight(
            in_c.data().data(), in_shape_,
            g.data().data(), out_shape_,
            g_w.data(), w_shape_,
            stride_, padding_, dilation_);

        std::vector<Tensor<T>> grads;
        grads.push_back(Tensor<T>::from_operation_result(in_shape_, std::move(g_in), false, nullptr));
        grads.push_back(Tensor<T>::from_operation_result(w_shape_, std::move(g_w), false, nullptr));
        if (has_bias_) {
            std::vector<T> g_b(static_cast<size_t>(w_shape_[1]));
            conv_backward_bias(g.data().data(), out_shape_, g_b.data());
            grads.push_back(Tensor<T>::from_operation_result(
                {w_shape_[1]}, std::move(g_b), false, nullptr));
        }
        return grads;
    }
};

} // namespace autograd
