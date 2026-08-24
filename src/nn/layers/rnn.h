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

template <typename T>
tensor::Tensor<T> linear(const tensor::Tensor<T>& x,
                         const tensor::Tensor<T>& w,
                         const std::optional<tensor::Tensor<T>>& b)
{
    auto y = ops::matmul(x, w.transpose());
    if (!b) return y;
    auto bb = b->unsqueeze(0).broadcast_to(y.shape()).contiguous();
    return ops::add(y, bb);
}

template <typename T>
tensor::Tensor<T> uniform_param(const std::vector<int64_t>& shape, int64_t hidden)
{
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

template <typename T>
CellWeights<T> make_cell(int64_t input_size, int64_t hidden_size, int64_t gate_mult, bool use_bias)
{
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

template <typename T>
void collect_cell(std::vector<tensor::Tensor<T>*>& out, CellWeights<T>& c)
{
    out.push_back(&c.weight_ih);
    out.push_back(&c.weight_hh);
    if (c.bias_ih) out.push_back(&(*c.bias_ih));
    if (c.bias_hh) out.push_back(&(*c.bias_hh));
}

template <typename T>
tensor::Tensor<T> rnn_step(const tensor::Tensor<T>& x,
                           const tensor::Tensor<T>& h,
                           const CellWeights<T>& w,
                           const std::string& nonlinearity)
{
    auto pre = ops::add(linear(x, w.weight_ih, w.bias_ih),
                        linear(h, w.weight_hh, w.bias_hh));
    if (nonlinearity == "relu")
        return ops::relu(pre);
    return ops::tanh(pre);
}

template <typename T>
std::pair<tensor::Tensor<T>, tensor::Tensor<T>> lstm_step(
    const tensor::Tensor<T>& x,
    const tensor::Tensor<T>& h,
    const tensor::Tensor<T>& c,
    const CellWeights<T>& w)
{
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

template <typename T>
tensor::Tensor<T> gru_step(const tensor::Tensor<T>& x,
                           const tensor::Tensor<T>& h,
                           const CellWeights<T>& w)
{
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

template <typename T>
tensor::Tensor<T> time_slice(const tensor::Tensor<T>& seq, int64_t t, bool batch_first)
{
    // seq-first (T, N, F) or batch-first (N, T, F)
    if (batch_first)
        return seq.narrow(1, t, 1).squeeze(1);
    return seq.narrow(0, t, 1).squeeze(0);
}

template <typename T>
tensor::Tensor<T> stack_time(const std::vector<tensor::Tensor<T>>& steps, bool batch_first)
{
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
public:
    std::vector<rnn_detail::CellWeights<T>> cells;          // forward, size num_layers
    std::vector<rnn_detail::CellWeights<T>> cells_reverse;  // empty if not bidirectional

    RNN(int64_t input_size, int64_t hidden_size, int64_t num_layers = 1,
        bool use_bias = true, bool batch_first = false, bool bidirectional = false,
        std::string nonlinearity = "tanh")
        : input_size_(input_size), hidden_size_(hidden_size), num_layers_(num_layers),
          batch_first_(batch_first), bidirectional_(bidirectional),
          nonlinearity_(std::move(nonlinearity))
    {
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

    std::pair<tensor::Tensor<T>, tensor::Tensor<T>>
    forward(const tensor::Tensor<T>& input) const
    {
        return forward(input, std::nullopt);
    }

    std::pair<tensor::Tensor<T>, tensor::Tensor<T>>
    forward(const tensor::Tensor<T>& input, const tensor::Tensor<T>& h0) const
    {
        return forward(input, std::optional<tensor::Tensor<T>>(h0));
    }

    std::pair<tensor::Tensor<T>, tensor::Tensor<T>>
    forward(const tensor::Tensor<T>& input,
            const std::optional<tensor::Tensor<T>>& h0) const
    {
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

    std::vector<tensor::Tensor<T>*> parameters() override
    {
        std::vector<tensor::Tensor<T>*> out;
        for (auto& c : cells) rnn_detail::collect_cell(out, c);
        for (auto& c : cells_reverse) rnn_detail::collect_cell(out, c);
        return out;
    }

private:
    int64_t input_size_, hidden_size_, num_layers_;
    bool batch_first_, bidirectional_;
    std::string nonlinearity_;
};

template <typename T>
class LSTM : public Layer<T> {
public:
    std::vector<rnn_detail::CellWeights<T>> cells;
    std::vector<rnn_detail::CellWeights<T>> cells_reverse;

    LSTM(int64_t input_size, int64_t hidden_size, int64_t num_layers = 1,
         bool use_bias = true, bool batch_first = false, bool bidirectional = false)
        : input_size_(input_size), hidden_size_(hidden_size), num_layers_(num_layers),
          batch_first_(batch_first), bidirectional_(bidirectional)
    {
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

    std::pair<tensor::Tensor<T>, std::pair<tensor::Tensor<T>, tensor::Tensor<T>>>
    forward(const tensor::Tensor<T>& input) const
    {
        return forward(input, std::nullopt, std::nullopt);
    }

    std::pair<tensor::Tensor<T>, std::pair<tensor::Tensor<T>, tensor::Tensor<T>>>
    forward(const tensor::Tensor<T>& input, const tensor::Tensor<T>& h0) const
    {
        return forward(input, std::optional<tensor::Tensor<T>>(h0), std::nullopt);
    }

    std::pair<tensor::Tensor<T>, std::pair<tensor::Tensor<T>, tensor::Tensor<T>>>
    forward(const tensor::Tensor<T>& input,
            const std::optional<tensor::Tensor<T>>& h0,
            const std::optional<tensor::Tensor<T>>& c0) const
    {
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

    std::vector<tensor::Tensor<T>*> parameters() override
    {
        std::vector<tensor::Tensor<T>*> out;
        for (auto& c : cells) rnn_detail::collect_cell(out, c);
        for (auto& c : cells_reverse) rnn_detail::collect_cell(out, c);
        return out;
    }

private:
    int64_t input_size_, hidden_size_, num_layers_;
    bool batch_first_, bidirectional_;
};

template <typename T>
class GRU : public Layer<T> {
public:
    std::vector<rnn_detail::CellWeights<T>> cells;
    std::vector<rnn_detail::CellWeights<T>> cells_reverse;

    GRU(int64_t input_size, int64_t hidden_size, int64_t num_layers = 1,
        bool use_bias = true, bool batch_first = false, bool bidirectional = false)
        : input_size_(input_size), hidden_size_(hidden_size), num_layers_(num_layers),
          batch_first_(batch_first), bidirectional_(bidirectional)
    {
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

    std::pair<tensor::Tensor<T>, tensor::Tensor<T>>
    forward(const tensor::Tensor<T>& input) const
    {
        return forward(input, std::nullopt);
    }

    std::pair<tensor::Tensor<T>, tensor::Tensor<T>>
    forward(const tensor::Tensor<T>& input, const tensor::Tensor<T>& h0) const
    {
        return forward(input, std::optional<tensor::Tensor<T>>(h0));
    }

    std::pair<tensor::Tensor<T>, tensor::Tensor<T>>
    forward(const tensor::Tensor<T>& input,
            const std::optional<tensor::Tensor<T>>& h0) const
    {
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

    std::vector<tensor::Tensor<T>*> parameters() override
    {
        std::vector<tensor::Tensor<T>*> out;
        for (auto& c : cells) rnn_detail::collect_cell(out, c);
        for (auto& c : cells_reverse) rnn_detail::collect_cell(out, c);
        return out;
    }

private:
    int64_t input_size_, hidden_size_, num_layers_;
    bool batch_first_, bidirectional_;
};

} // namespace nn
