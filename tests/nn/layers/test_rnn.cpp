#include <cmath>
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

TEST_CASE("RNN batch_first and bidirectional") {
    nn::RNN<float> rnn(3, 4, 1, true, true, false);
    tensor::Tensor<float> x({2, 5, 3}, std::vector<float>(30, 0.1f), false);
    auto [out, hn] = rnn.forward(x);
    CHECK(out.shape() == std::vector<int64_t>{2, 5, 4});
    CHECK(hn.shape() == std::vector<int64_t>{1, 2, 4});

    nn::RNN<float> bi(3, 4, 1, true, false, true);
    tensor::Tensor<float> xs({5, 2, 3}, std::vector<float>(30, 0.1f), false);
    auto [bout, bhn] = bi.forward(xs);
    CHECK(bout.shape() == std::vector<int64_t>{5, 2, 8});
    CHECK(bhn.shape() == std::vector<int64_t>{2, 2, 4});
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
