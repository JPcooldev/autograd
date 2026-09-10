/*
 * RNN / LSTM / GRU: init bounds, layouts, and one numeric tanh step.
 *
 * - RNN params in U(-1/sqrt(H), 1/sqrt(H))
 * - seq-first output and h_n; backward into weight_ih
 * - batch_first layout
 * - bidirectional concat and `_reverse` parameter names
 * - batch_first and bidirectional together
 * - T=1 known weights match tanh(x)
 * - LSTM / GRU output and state sizes; LSTM backward
 * - num_layers stacks h_n
 */

#include <cmath>
#include <string>
#include <vector>

#include "../../doctest/doctest.h"

#include "../../../src/autograd/autograd.h"
#include "../../../src/nn/layers/rnn.h"

TEST_CASE("RNN params are U(-1/sqrt(H), 1/sqrt(H))") {
    nn::RNN<float> rnn(3, 8);
    const float bound = 1.f / std::sqrt(8.f);
    REQUIRE(rnn.parameters().size() == 4);
    for (auto* p : rnn.parameters()) {
        for (float v : p->data())
            CHECK(std::abs(v) <= bound + 1e-5f);
    }
}

TEST_CASE("RNN seq-first shapes") {
    nn::RNN<float> rnn(3, 4);
    tensor::Tensor<float> x({2, 1, 3}, std::vector<float>(6, 0.2f), true);
    auto [out, hn] = rnn.forward(x);
    CHECK(out.shape() == std::vector<int64_t>{2, 1, 4});
    CHECK(hn.shape() == std::vector<int64_t>{1, 1, 4});
    out.sum().backward();
    REQUIRE(rnn.cells[0].weight_ih.grad() != nullptr);
}

TEST_CASE("RNN batch_first layout") {
    nn::RNN<float> rnn(3, 4, 1, true, true, false);
    tensor::Tensor<float> x({2, 5, 3}, std::vector<float>(30, 0.1f), false);
    auto [out, hn] = rnn.forward(x);
    CHECK(out.shape() == std::vector<int64_t>{2, 5, 4});
    CHECK(hn.shape() == std::vector<int64_t>{1, 2, 4});
}

TEST_CASE("RNN bidirectional seq-first concatenates hidden dim") {
    nn::RNN<float> bi(3, 4, 1, true, false, true);
    tensor::Tensor<float> xs({5, 2, 3}, std::vector<float>(30, 0.1f), false);
    auto [bout, bhn] = bi.forward(xs);
    CHECK(bout.shape() == std::vector<int64_t>{5, 2, 8});
    CHECK(bhn.shape() == std::vector<int64_t>{2, 2, 4});
    bool saw_reverse = false;
    for (auto& kv : bi.named_parameters())
        if (kv.first.find("_reverse") != std::string::npos)
            saw_reverse = true;
    CHECK(saw_reverse);
}

TEST_CASE("RNN batch_first and bidirectional together") {
    nn::RNN<float> rnn(3, 4, 1, true, true, true);
    tensor::Tensor<float> x({2, 5, 3}, std::vector<float>(30, 0.1f), false);
    auto [out, hn] = rnn.forward(x);
    CHECK(out.shape() == std::vector<int64_t>{2, 5, 8});
    CHECK(hn.shape() == std::vector<int64_t>{2, 2, 4});
}

TEST_CASE("RNN T=1 known weights matches tanh") {
    nn::RNN<float> rnn(1, 1, 1, false, false, false);
    rnn.cells[0].weight_ih.data() = {1.f};
    rnn.cells[0].weight_hh.data() = {0.f};
    tensor::Tensor<float> x({1, 1, 1}, std::vector<float>{2.f}, true);
    auto [out, hn] = rnn.forward(x);
    CHECK(out.data()[0] == doctest::Approx(std::tanh(2.f)));
    CHECK(hn.data()[0] == doctest::Approx(std::tanh(2.f)));
    out.sum().backward();
    REQUIRE(rnn.cells[0].weight_ih.grad() != nullptr);
}

TEST_CASE("LSTM and GRU shapes") {
    nn::LSTM<float> lstm(3, 4);
    nn::GRU<float> gru(3, 4);
    tensor::Tensor<float> x({2, 1, 3}, std::vector<float>(6, 0.3f), true);
    auto [lo, hc] = lstm.forward(x);
    CHECK(lo.shape() == std::vector<int64_t>{2, 1, 4});
    CHECK(hc.first.shape() == std::vector<int64_t>{1, 1, 4});
    CHECK(hc.second.shape() == std::vector<int64_t>{1, 1, 4});
    auto [go, ghn] = gru.forward(x);
    CHECK(go.shape() == std::vector<int64_t>{2, 1, 4});
    CHECK(ghn.shape() == std::vector<int64_t>{1, 1, 4});
    lo.sum().backward();
    REQUIRE(lstm.cells[0].weight_ih.grad() != nullptr);
}

TEST_CASE("RNN num_layers stacks hidden size") {
    nn::RNN<float> rnn(3, 4, 2);
    tensor::Tensor<float> x({3, 2, 3}, std::vector<float>(18, 0.1f), false);
    auto [out, hn] = rnn.forward(x);
    CHECK(out.shape() == std::vector<int64_t>{3, 2, 4});
    CHECK(hn.shape() == std::vector<int64_t>{2, 2, 4});
    CHECK(rnn.parameters().size() == 8);
}
