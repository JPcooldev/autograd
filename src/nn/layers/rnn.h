#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "../../tensor/tensor.h"
#include "layer.h"
#include "../../ops/elementwise_ops.h"
#include "../../ops/linalg_ops.h"
#include "../../ops/shape_ops.h"

namespace nn {

namespace rnn_detail {

/**
 * Affine map `x @ W.T [+ b]` used inside recurrent cells.
 * `matmul`s `x` with `w.transpose()`; if `b` is set, broadcasts it over the batch and adds.
 *
 * @param x Rank-2 input `{batch, in}`.
 * @param w Weight `{out, in}`.
 * @param b Optional bias `{out}`.
 * @return Rank-2 `{batch, out}`.
 */
template <typename T>
tensor::Tensor<T> linear(const tensor::Tensor<T>& x,
                         const tensor::Tensor<T>& w,
                         const std::optional<tensor::Tensor<T>>& b) {
    auto y = ops::matmul(x, w.transpose());
    if (!b)
        return y;
    auto bb = b->unsqueeze(0).broadcast_to(y.shape()).contiguous();
    return ops::add(y, bb);
}

/**
 * Allocate a learnable tensor with U(-1/√H, 1/√H).
 *
 * @param shape Tensor shape.
 * @param hidden Hidden size H used for the uniform bound.
 * @return New tensor with `requires_grad == true`.
 */
template <typename T>
tensor::Tensor<T> uniform_param(const std::vector<int64_t>& shape, int64_t hidden) {
    const T bound = static_cast<T>(1.0 / std::sqrt(static_cast<double>(hidden)));
    return tensor::Tensor<T>::uniform(shape, -bound, bound, true);
}

template <typename T>
struct CellWeights {
    tensor::Tensor<T> weight_ih;
    tensor::Tensor<T> weight_hh;
    std::optional<tensor::Tensor<T>> bias_ih;
    std::optional<tensor::Tensor<T>> bias_hh;
};

/**
 * Build one recurrent cell's weights and optional biases.
 * `weight_ih` is `{gate_mult * H, input_size}`, `weight_hh` is `{gate_mult * H, H}`;
 * biases match `{gate_mult * H}` when `use_bias` is true. All use `uniform_param`.
 *
 * @param input_size Input feature size for this layer.
 * @param hidden_size Hidden size H.
 * @param gate_mult 1 (RNN), 4 (LSTM), or 3 (GRU).
 * @param use_bias Whether to allocate `bias_ih` and `bias_hh`.
 * @return Populated `CellWeights`.
 */
template <typename T>
CellWeights<T> make_cell(int64_t input_size, int64_t hidden_size, int64_t gate_mult, bool use_bias) {
    std::optional<tensor::Tensor<T>> b_ih;
    std::optional<tensor::Tensor<T>> b_hh;
    if (use_bias) {
        b_ih = uniform_param<T>({gate_mult * hidden_size}, hidden_size);
        b_hh = uniform_param<T>({gate_mult * hidden_size}, hidden_size);
    }
    return CellWeights<T>{
        uniform_param<T>({gate_mult * hidden_size, input_size}, hidden_size),
        uniform_param<T>({gate_mult * hidden_size, hidden_size}, hidden_size),
        std::move(b_ih),
        std::move(b_hh)
    };
}

/**
 * Append a cell's learnable tensors to `out` with PyTorch-style names.
 * Names are `weight_ih_l{layer}`, `weight_hh_l{layer}`, and optional biases;
 * reverse cells add the `_reverse` suffix.
 *
 * @param out Named parameter list being collected.
 * @param c Cell whose tensors are appended.
 * @param layer Layer index (0-based).
 * @param reverse True for the reverse direction of a bidirectional RNN.
 */
template <typename T>
void collect_cell_named(
    NamedTensorList<T>& out,
    CellWeights<T>& c,
    int64_t layer,
    bool reverse
) {
    const std::string suffix = reverse
        ? ("_l" + std::to_string(layer) + "_reverse")
        : ("_l" + std::to_string(layer));
    out.emplace_back("weight_ih" + suffix, &c.weight_ih);
    out.emplace_back("weight_hh" + suffix, &c.weight_hh);
    if (c.bias_ih)
        out.emplace_back("bias_ih" + suffix, &(*c.bias_ih));
    if (c.bias_hh)
        out.emplace_back("bias_hh" + suffix, &(*c.bias_hh));
}

/**
 * One Elman RNN step: tanh or ReLU of the pre-activation.
 * Pre-activation is `linear(x, W_ih, b_ih) + linear(h, W_hh, b_hh)`. `"relu"` selects ReLU; anything else uses tanh.
 *
 * @param x Input at this time `{batch, in}`.
 * @param h Hidden state `{batch, H}`.
 * @param w Cell weights.
 * @param nonlinearity `"relu"` or `"tanh"`.
 * @return Next hidden state `{batch, H}`.
 */
template <typename T>
tensor::Tensor<T> rnn_step(const tensor::Tensor<T>& x,
                           const tensor::Tensor<T>& h,
                           const CellWeights<T>& w,
                           const std::string& nonlinearity) {
    auto pre = ops::add(linear(x, w.weight_ih, w.bias_ih),
                        linear(h, w.weight_hh, w.bias_hh));
    if (nonlinearity == "relu")
        return ops::relu(pre);
    return ops::tanh(pre);
}

/**
 * One LSTM step (input, forget, cell, output gates).
 * Splits the 4H pre-activation along the last axis into i, f, n, o; updates
 * `c_n = f⊙c + i⊙n` and `h_n = o⊙tanh(c_n)`.
 *
 * @param x Input at this time `{batch, in}`.
 * @param h Hidden state `{batch, H}`.
 * @param c Cell state `{batch, H}`.
 * @param w Cell weights with `gate_mult == 4`.
 * @return Pair `(h_n, c_n)`.
 */
template <typename T>
std::pair<tensor::Tensor<T>, tensor::Tensor<T>> lstm_step(
    const tensor::Tensor<T>& x,
    const tensor::Tensor<T>& h,
    const tensor::Tensor<T>& c,
    const CellWeights<T>& w) {
    const int64_t H = h.shape().back();
    auto g = ops::add(linear(x, w.weight_ih, w.bias_ih),
                      linear(h, w.weight_hh, w.bias_hh));
    auto i = ops::sigmoid(g.narrow(-1, 0, H));
    auto f = ops::sigmoid(g.narrow(-1, H, H));
    auto n = ops::tanh(g.narrow(-1, 2 * H, H));
    auto o = ops::sigmoid(g.narrow(-1, 3 * H, H));
    auto c_n = ops::add(ops::multiply(f, c), ops::multiply(i, n));
    auto h_n = ops::multiply(o, ops::tanh(c_n));
    return {h_n, c_n};
}

/**
 * One GRU step (reset, update, new gates).
 * Uses the PyTorch split of `W_ih`/`W_hh` into r, z, n; returns `(1-z)⊙n + z⊙h`.
 *
 * @param x Input at this time `{batch, in}`.
 * @param h Hidden state `{batch, H}`.
 * @param w Cell weights with `gate_mult == 3`.
 * @return Next hidden state `{batch, H}`.
 */
template <typename T>
tensor::Tensor<T> gru_step(const tensor::Tensor<T>& x,
                           const tensor::Tensor<T>& h,
                           const CellWeights<T>& w) {
    const int64_t H = h.shape().back();
    auto gi = linear(x, w.weight_ih, w.bias_ih);
    auto gh = linear(h, w.weight_hh, w.bias_hh);
    auto r = ops::sigmoid(ops::add(gi.narrow(-1, 0, H), gh.narrow(-1, 0, H)));
    auto z = ops::sigmoid(ops::add(gi.narrow(-1, H, H), gh.narrow(-1, H, H)));
    auto n = ops::tanh(ops::add(gi.narrow(-1, 2 * H, H),
                                ops::multiply(r, gh.narrow(-1, 2 * H, H))));
    auto ones = tensor::Tensor<T>::ones(z.shape(), false);
    auto omz = ops::subtract(ones, z);
    return ops::add(ops::multiply(omz, n), ops::multiply(z, h));
}

/**
 * Extract the feature slice at time `t`.
 * Seq-first `(T, N, F)` uses dim 0; batch-first `(N, T, F)` uses dim 1.
 *
 * @param seq Rank-3 sequence tensor.
 * @param t Time index.
 * @param batch_first Layout flag.
 * @return Rank-2 `{N, F}` slice.
 */
template <typename T>
tensor::Tensor<T> time_slice(const tensor::Tensor<T>& seq, int64_t t, bool batch_first) {
    if (batch_first)
        return seq.narrow(1, t, 1).squeeze(1);
    return seq.narrow(0, t, 1).squeeze(0);
}

/**
 * Stack per-timestep hidden states along the time axis.
 * Unsqueezes each step on dim 0 (seq-first) or dim 1 (batch-first) and `cat`s.
 *
 * @param steps Hidden tensors `{N, H}` in time order.
 * @param batch_first Layout flag.
 * @return Rank-3 sequence of hiddens.
 */
template <typename T>
tensor::Tensor<T> stack_time(const std::vector<tensor::Tensor<T>>& steps, bool batch_first) {
    std::vector<tensor::Tensor<T>> unsqueezed;
    unsqueezed.reserve(steps.size());
    const int64_t dim = batch_first ? 1 : 0;
    for (const auto& s : steps)
        unsqueezed.push_back(s.unsqueeze(dim));
    return ops::cat(unsqueezed, dim);
}

} // namespace rnn_detail

template <typename T>
class RNN : public Layer<T> {
private:
    int64_t input_size_, hidden_size_, num_layers_;
    bool batch_first_, bidirectional_;
    std::string nonlinearity_;

public:
    std::vector<rnn_detail::CellWeights<T>> cells;          // forward, size num_layers
    std::vector<rnn_detail::CellWeights<T>> cells_reverse;  // empty if not bidirectional

    /**
     * Construct a stacked Elman RNN.
     * Builds `num_layers` cells (`gate_mult == 1`); if bidirectional, also reverse cells.
     * Layer l>0 input size is `hidden_size * num_directions`. Weights and biases use U(-1/√H, 1/√H).
     *
     * @param input_size Feature size of the input sequence.
     * @param hidden_size Hidden size H.
     * @param num_layers Number of stacked layers. Default 1.
     * @param use_bias Whether cells have biases. Default true.
     * @param batch_first If true, input layout is `(N, T, F)`; else `(T, N, F)`. Default false.
     * @param bidirectional If true, each layer has a reverse cell. Default false.
     * @param nonlinearity `"tanh"` or `"relu"`. Default `"tanh"`.
     *
     * @throws std::invalid_argument if `nonlinearity` is not `"tanh"` or `"relu"`.
     * @throws std::invalid_argument if `num_layers < 1`.
     */
    RNN(int64_t input_size, int64_t hidden_size, int64_t num_layers = 1,
        bool use_bias = true, bool batch_first = false, bool bidirectional = false,
        std::string nonlinearity = "tanh")
        : input_size_(input_size), hidden_size_(hidden_size), num_layers_(num_layers),
          batch_first_(batch_first), bidirectional_(bidirectional),
          nonlinearity_(std::move(nonlinearity)) {
        if (nonlinearity_ != "tanh" && nonlinearity_ != "relu")
            throw std::invalid_argument("RNN: nonlinearity must be \"tanh\" or \"relu\"");
        if (num_layers < 1)
            throw std::invalid_argument("RNN: num_layers must be >= 1");
        int64_t in = input_size;
        const int64_t dirs = bidirectional ? 2 : 1;
        for (int64_t l = 0; l < num_layers; ++l) {
            cells.push_back(rnn_detail::make_cell<T>(in, hidden_size, 1, use_bias));
            if (bidirectional)
                cells_reverse.push_back(rnn_detail::make_cell<T>(in, hidden_size, 1, use_bias));
            in = hidden_size * dirs;
        }
    }

    /**
     * Run the RNN with zero initial hidden state.
     * Forwards to `forward(input, nullopt)`.
     *
     * @param input Rank-3 sequence `(T, N, F)` or `(N, T, F)` if `batch_first`.
     * @return `(output, h_n)` with `h_n` shape `(num_layers * num_directions, N, H)`.
     *
     * @throws std::invalid_argument if `input` is not rank 3.
     */
    std::pair<tensor::Tensor<T>, tensor::Tensor<T>>
    forward(const tensor::Tensor<T>& input) const {
        return forward(input, std::nullopt);
    }

    /**
     * Run the RNN with an explicit initial hidden state.
     * Wraps `h0` in `optional` and forwards to the three-argument `forward`.
     *
     * @param input Rank-3 sequence.
     * @param h0 Initial hidden `(num_layers * num_directions, N, H)`.
     * @return `(output, h_n)`.
     *
     * @throws std::invalid_argument if `input` is not rank 3.
     * @throws std::invalid_argument if `h0` cannot be narrowed to each layer's hidden (out of range).
     */
    std::pair<tensor::Tensor<T>, tensor::Tensor<T>>
    forward(const tensor::Tensor<T>& input, const tensor::Tensor<T>& h0) const {
        return forward(input, std::optional<tensor::Tensor<T>>(h0));
    }

    /**
     * Unroll the stacked (optionally bidirectional) Elman RNN.
     * Each layer runs forward in time; reverse direction runs backward then concatenates on the
     * feature axis. Output width is `H` or `2H` if bidirectional. Missing `h0` is zeros `{N, H}`.
     *
     * @param input Rank-3 sequence `(T, N, F)` or `(N, T, F)` if `batch_first`.
     * @param h0 Optional initial hidden `(num_layers * num_directions, N, H)`.
     * @return `(output, h_n)` where `output` matches input layout with last dim `H` or `2H`.
     *
     * @throws std::invalid_argument if `input` is not rank 3.
     * @throws std::invalid_argument if `h0` is set but slice indices are out of range.
     */
    std::pair<tensor::Tensor<T>, tensor::Tensor<T>>
    forward(const tensor::Tensor<T>& input,
            const std::optional<tensor::Tensor<T>>& h0) const {
        if (input.rank() != 3)
            throw std::invalid_argument("RNN: input must be 3-D");
        const int64_t seq_dim = batch_first_ ? 1 : 0;
        const int64_t batch_dim = batch_first_ ? 0 : 1;
        const int64_t Tlen = input.shape()[static_cast<size_t>(seq_dim)];
        const int64_t B = input.shape()[static_cast<size_t>(batch_dim)];
        const int64_t dirs = bidirectional_ ? 2 : 1;

        auto seq = input;
        std::vector<tensor::Tensor<T>> h_n_layers;
        for (int64_t l = 0; l < num_layers_; ++l) {
            auto h_fwd = h0
                ? h0->narrow(0, l * dirs, 1).squeeze(0)
                : tensor::Tensor<T>::zeros({B, hidden_size_}, false);
            std::vector<tensor::Tensor<T>> fwd_steps;
            fwd_steps.reserve(static_cast<size_t>(Tlen));
            for (int64_t t = 0; t < Tlen; ++t) {
                auto x_t = rnn_detail::time_slice(seq, t, batch_first_);
                h_fwd = rnn_detail::rnn_step(x_t, h_fwd, cells[static_cast<size_t>(l)], nonlinearity_);
                fwd_steps.push_back(h_fwd);
            }
            h_n_layers.push_back(h_fwd.unsqueeze(0));
            tensor::Tensor<T> layer_out = rnn_detail::stack_time(fwd_steps, batch_first_);
            if (bidirectional_) {
                auto h_rev = h0
                    ? h0->narrow(0, l * dirs + 1, 1).squeeze(0)
                    : tensor::Tensor<T>::zeros({B, hidden_size_}, false);
                std::vector<tensor::Tensor<T>> rev_acc;
                rev_acc.reserve(static_cast<size_t>(Tlen));
                for (int64_t t = Tlen - 1; t >= 0; --t) {
                    auto x_t = rnn_detail::time_slice(seq, t, batch_first_);
                    h_rev = rnn_detail::rnn_step(
                        x_t, h_rev, cells_reverse[static_cast<size_t>(l)], nonlinearity_);
                    rev_acc.push_back(h_rev);
                }
                std::reverse(rev_acc.begin(), rev_acc.end());
                h_n_layers.push_back(h_rev.unsqueeze(0));
                auto rev_out = rnn_detail::stack_time(rev_acc, batch_first_);
                layer_out = ops::cat({layer_out, rev_out}, 2);
            }
            seq = layer_out;
        }
        return {seq, ops::cat(h_n_layers, 0)};
    }

    /**
     * Collect named weights and biases from forward and reverse cells.
     * Forward cells use `_l{i}` names; reverse cells use `_l{i}_reverse`.
     *
     * @return Named tensors in `cells` then `cells_reverse`.
     */
    NamedTensorList<T> named_parameters() override {
        NamedTensorList<T> out;
        for (size_t i = 0; i < cells.size(); ++i)
            rnn_detail::collect_cell_named(out, cells[i], static_cast<int64_t>(i), false);
        for (size_t i = 0; i < cells_reverse.size(); ++i)
            rnn_detail::collect_cell_named(out, cells_reverse[i], static_cast<int64_t>(i), true);
        return out;
    }
};

template <typename T>
class LSTM : public Layer<T> {
private:
    int64_t input_size_, hidden_size_, num_layers_;
    bool batch_first_, bidirectional_;

public:
    std::vector<rnn_detail::CellWeights<T>> cells;
    std::vector<rnn_detail::CellWeights<T>> cells_reverse;

    /**
     * Construct a stacked LSTM.
     * Builds `num_layers` cells with `gate_mult == 4` (i, f, n, o); reverse cells if bidirectional.
     * All parameters use U(-1/√H, 1/√H).
     *
     * @param input_size Feature size of the input sequence.
     * @param hidden_size Hidden size H.
     * @param num_layers Number of stacked layers. Default 1.
     * @param use_bias Whether cells have biases. Default true.
     * @param batch_first If true, input layout is `(N, T, F)`; else `(T, N, F)`. Default false.
     * @param bidirectional If true, each layer has a reverse cell. Default false.
     *
     * @throws std::invalid_argument if `num_layers < 1`.
     */
    LSTM(int64_t input_size, int64_t hidden_size, int64_t num_layers = 1,
         bool use_bias = true, bool batch_first = false, bool bidirectional = false)
        : input_size_(input_size), hidden_size_(hidden_size), num_layers_(num_layers),
          batch_first_(batch_first), bidirectional_(bidirectional) {
        if (num_layers < 1)
            throw std::invalid_argument("LSTM: num_layers must be >= 1");
        int64_t in = input_size;
        const int64_t dirs = bidirectional ? 2 : 1;
        for (int64_t l = 0; l < num_layers; ++l) {
            cells.push_back(rnn_detail::make_cell<T>(in, hidden_size, 4, use_bias));
            if (bidirectional)
                cells_reverse.push_back(rnn_detail::make_cell<T>(in, hidden_size, 4, use_bias));
            in = hidden_size * dirs;
        }
    }

    /**
     * Run the LSTM with zero initial hidden and cell states.
     * Forwards to `forward(input, nullopt, nullopt)`.
     *
     * @param input Rank-3 sequence.
     * @return `(output, (h_n, c_n))`.
     *
     * @throws std::invalid_argument if `input` is not rank 3.
     */
    std::pair<tensor::Tensor<T>, std::pair<tensor::Tensor<T>, tensor::Tensor<T>>>
    forward(const tensor::Tensor<T>& input) const {
        return forward(input, std::nullopt, std::nullopt);
    }

    /**
     * Run the LSTM with explicit `h0` and zero `c0`.
     * Forwards to `forward(input, h0, nullopt)`.
     *
     * @param input Rank-3 sequence.
     * @param h0 Initial hidden `(num_layers * num_directions, N, H)`.
     * @return `(output, (h_n, c_n))`.
     *
     * @throws std::invalid_argument if `input` is not rank 3.
     * @throws std::invalid_argument if `h0` cannot be narrowed to each layer's hidden.
     */
    std::pair<tensor::Tensor<T>, std::pair<tensor::Tensor<T>, tensor::Tensor<T>>>
    forward(const tensor::Tensor<T>& input, const tensor::Tensor<T>& h0) const {
        return forward(input, std::optional<tensor::Tensor<T>>(h0), std::nullopt);
    }

    /**
     * Unroll the stacked (optionally bidirectional) LSTM.
     * Each layer updates `(h, c)` with `lstm_step`; reverse direction concatenates on the feature axis.
     * Missing `h0`/`c0` are zeros `{N, H}`.
     *
     * @param input Rank-3 sequence `(T, N, F)` or `(N, T, F)` if `batch_first`.
     * @param h0 Optional initial hidden `(num_layers * num_directions, N, H)`.
     * @param c0 Optional initial cell, same shape as `h0`.
     * @return `(output, (h_n, c_n))` with `h_n`/`c_n` stacked on dim 0.
     *
     * @throws std::invalid_argument if `input` is not rank 3.
     * @throws std::invalid_argument if `h0` or `c0` is set but slice indices are out of range.
     */
    std::pair<tensor::Tensor<T>, std::pair<tensor::Tensor<T>, tensor::Tensor<T>>>
    forward(const tensor::Tensor<T>& input,
            const std::optional<tensor::Tensor<T>>& h0,
            const std::optional<tensor::Tensor<T>>& c0) const {
        if (input.rank() != 3)
            throw std::invalid_argument("LSTM: input must be 3-D");
        const int64_t seq_dim = batch_first_ ? 1 : 0;
        const int64_t batch_dim = batch_first_ ? 0 : 1;
        const int64_t Tlen = input.shape()[static_cast<size_t>(seq_dim)];
        const int64_t B = input.shape()[static_cast<size_t>(batch_dim)];
        const int64_t dirs = bidirectional_ ? 2 : 1;

        auto seq = input;
        std::vector<tensor::Tensor<T>> h_n_layers, c_n_layers;
        for (int64_t l = 0; l < num_layers_; ++l) {
            auto h = h0 ? h0->narrow(0, l * dirs, 1).squeeze(0)
                        : tensor::Tensor<T>::zeros({B, hidden_size_}, false);
            auto c = c0 ? c0->narrow(0, l * dirs, 1).squeeze(0)
                        : tensor::Tensor<T>::zeros({B, hidden_size_}, false);
            std::vector<tensor::Tensor<T>> fwd_steps;
            fwd_steps.reserve(static_cast<size_t>(Tlen));
            for (int64_t t = 0; t < Tlen; ++t) {
                auto x_t = rnn_detail::time_slice(seq, t, batch_first_);
                auto hc = rnn_detail::lstm_step(x_t, h, c, cells[static_cast<size_t>(l)]);
                h = hc.first; c = hc.second;
                fwd_steps.push_back(h);
            }
            h_n_layers.push_back(h.unsqueeze(0));
            c_n_layers.push_back(c.unsqueeze(0));
            tensor::Tensor<T> layer_out = rnn_detail::stack_time(fwd_steps, batch_first_);
            if (bidirectional_) {
                auto hr = h0 ? h0->narrow(0, l * dirs + 1, 1).squeeze(0)
                             : tensor::Tensor<T>::zeros({B, hidden_size_}, false);
                auto cr = c0 ? c0->narrow(0, l * dirs + 1, 1).squeeze(0)
                             : tensor::Tensor<T>::zeros({B, hidden_size_}, false);
                std::vector<tensor::Tensor<T>> rev_acc;
                rev_acc.reserve(static_cast<size_t>(Tlen));
                for (int64_t t = Tlen - 1; t >= 0; --t) {
                    auto x_t = rnn_detail::time_slice(seq, t, batch_first_);
                    auto hc = rnn_detail::lstm_step(
                        x_t, hr, cr, cells_reverse[static_cast<size_t>(l)]);
                    hr = hc.first; cr = hc.second;
                    rev_acc.push_back(hr);
                }
                std::reverse(rev_acc.begin(), rev_acc.end());
                h_n_layers.push_back(hr.unsqueeze(0));
                c_n_layers.push_back(cr.unsqueeze(0));
                layer_out = ops::cat({layer_out, rnn_detail::stack_time(rev_acc, batch_first_)}, 2);
            }
            seq = layer_out;
        }
        return {seq, {ops::cat(h_n_layers, 0), ops::cat(c_n_layers, 0)}};
    }

    /**
     * Collect named weights and biases from forward and reverse cells.
     * Forward cells use `_l{i}` names; reverse cells use `_l{i}_reverse`.
     *
     * @return Named tensors in `cells` then `cells_reverse`.
     */
    NamedTensorList<T> named_parameters() override {
        NamedTensorList<T> out;
        for (size_t i = 0; i < cells.size(); ++i)
            rnn_detail::collect_cell_named(out, cells[i], static_cast<int64_t>(i), false);
        for (size_t i = 0; i < cells_reverse.size(); ++i)
            rnn_detail::collect_cell_named(out, cells_reverse[i], static_cast<int64_t>(i), true);
        return out;
    }
};

template <typename T>
class GRU : public Layer<T> {
private:
    int64_t input_size_, hidden_size_, num_layers_;
    bool batch_first_, bidirectional_;

public:
    std::vector<rnn_detail::CellWeights<T>> cells;
    std::vector<rnn_detail::CellWeights<T>> cells_reverse;

    /**
     * Construct a stacked GRU.
     * Builds `num_layers` cells with `gate_mult == 3` (r, z, n); reverse cells if bidirectional.
     * All parameters use U(-1/√H, 1/√H).
     *
     * @param input_size Feature size of the input sequence.
     * @param hidden_size Hidden size H.
     * @param num_layers Number of stacked layers. Default 1.
     * @param use_bias Whether cells have biases. Default true.
     * @param batch_first If true, input layout is `(N, T, F)`; else `(T, N, F)`. Default false.
     * @param bidirectional If true, each layer has a reverse cell. Default false.
     *
     * @throws std::invalid_argument if `num_layers < 1`.
     */
    GRU(int64_t input_size, int64_t hidden_size, int64_t num_layers = 1,
        bool use_bias = true, bool batch_first = false, bool bidirectional = false)
        : input_size_(input_size), hidden_size_(hidden_size), num_layers_(num_layers),
          batch_first_(batch_first), bidirectional_(bidirectional) {
        if (num_layers < 1)
            throw std::invalid_argument("GRU: num_layers must be >= 1");
        int64_t in = input_size;
        const int64_t dirs = bidirectional ? 2 : 1;
        for (int64_t l = 0; l < num_layers; ++l) {
            cells.push_back(rnn_detail::make_cell<T>(in, hidden_size, 3, use_bias));
            if (bidirectional)
                cells_reverse.push_back(rnn_detail::make_cell<T>(in, hidden_size, 3, use_bias));
            in = hidden_size * dirs;
        }
    }

    /**
     * Run the GRU with zero initial hidden state.
     * Forwards to `forward(input, nullopt)`.
     *
     * @param input Rank-3 sequence.
     * @return `(output, h_n)`.
     *
     * @throws std::invalid_argument if `input` is not rank 3.
     */
    std::pair<tensor::Tensor<T>, tensor::Tensor<T>>
    forward(const tensor::Tensor<T>& input) const {
        return forward(input, std::nullopt);
    }

    /**
     * Run the GRU with an explicit initial hidden state.
     * Wraps `h0` in `optional` and forwards to the three-argument `forward`.
     *
     * @param input Rank-3 sequence.
     * @param h0 Initial hidden `(num_layers * num_directions, N, H)`.
     * @return `(output, h_n)`.
     *
     * @throws std::invalid_argument if `input` is not rank 3.
     * @throws std::invalid_argument if `h0` cannot be narrowed to each layer's hidden.
     */
    std::pair<tensor::Tensor<T>, tensor::Tensor<T>>
    forward(const tensor::Tensor<T>& input, const tensor::Tensor<T>& h0) const {
        return forward(input, std::optional<tensor::Tensor<T>>(h0));
    }

    /**
     * Unroll the stacked (optionally bidirectional) GRU.
     * Each layer runs `gru_step` in time; reverse direction concatenates on the feature axis.
     * Missing `h0` is zeros `{N, H}`.
     *
     * @param input Rank-3 sequence `(T, N, F)` or `(N, T, F)` if `batch_first`.
     * @param h0 Optional initial hidden `(num_layers * num_directions, N, H)`.
     * @return `(output, h_n)` with `h_n` stacked on dim 0.
     *
     * @throws std::invalid_argument if `input` is not rank 3.
     * @throws std::invalid_argument if `h0` is set but slice indices are out of range.
     */
    std::pair<tensor::Tensor<T>, tensor::Tensor<T>>
    forward(const tensor::Tensor<T>& input,
            const std::optional<tensor::Tensor<T>>& h0) const {
        if (input.rank() != 3)
            throw std::invalid_argument("GRU: input must be 3-D");
        const int64_t seq_dim = batch_first_ ? 1 : 0;
        const int64_t batch_dim = batch_first_ ? 0 : 1;
        const int64_t Tlen = input.shape()[static_cast<size_t>(seq_dim)];
        const int64_t B = input.shape()[static_cast<size_t>(batch_dim)];
        const int64_t dirs = bidirectional_ ? 2 : 1;

        auto seq = input;
        std::vector<tensor::Tensor<T>> h_n_layers;
        for (int64_t l = 0; l < num_layers_; ++l) {
            auto h = h0 ? h0->narrow(0, l * dirs, 1).squeeze(0)
                        : tensor::Tensor<T>::zeros({B, hidden_size_}, false);
            std::vector<tensor::Tensor<T>> fwd_steps;
            fwd_steps.reserve(static_cast<size_t>(Tlen));
            for (int64_t t = 0; t < Tlen; ++t) {
                auto x_t = rnn_detail::time_slice(seq, t, batch_first_);
                h = rnn_detail::gru_step(x_t, h, cells[static_cast<size_t>(l)]);
                fwd_steps.push_back(h);
            }
            h_n_layers.push_back(h.unsqueeze(0));
            tensor::Tensor<T> layer_out = rnn_detail::stack_time(fwd_steps, batch_first_);
            if (bidirectional_) {
                auto hr = h0 ? h0->narrow(0, l * dirs + 1, 1).squeeze(0)
                             : tensor::Tensor<T>::zeros({B, hidden_size_}, false);
                std::vector<tensor::Tensor<T>> rev_acc;
                rev_acc.reserve(static_cast<size_t>(Tlen));
                for (int64_t t = Tlen - 1; t >= 0; --t) {
                    auto x_t = rnn_detail::time_slice(seq, t, batch_first_);
                    hr = rnn_detail::gru_step(x_t, hr, cells_reverse[static_cast<size_t>(l)]);
                    rev_acc.push_back(hr);
                }
                std::reverse(rev_acc.begin(), rev_acc.end());
                h_n_layers.push_back(hr.unsqueeze(0));
                layer_out = ops::cat({layer_out, rnn_detail::stack_time(rev_acc, batch_first_)}, 2);
            }
            seq = layer_out;
        }
        return {seq, ops::cat(h_n_layers, 0)};
    }

    /**
     * Collect named weights and biases from forward and reverse cells.
     * Forward cells use `_l{i}` names; reverse cells use `_l{i}_reverse`.
     *
     * @return Named tensors in `cells` then `cells_reverse`.
     */
    NamedTensorList<T> named_parameters() override {
        NamedTensorList<T> out;
        for (size_t i = 0; i < cells.size(); ++i)
            rnn_detail::collect_cell_named(out, cells[i], static_cast<int64_t>(i), false);
        for (size_t i = 0; i < cells_reverse.size(); ++i)
            rnn_detail::collect_cell_named(out, cells_reverse[i], static_cast<int64_t>(i), true);
        return out;
    }
};

} // namespace nn
