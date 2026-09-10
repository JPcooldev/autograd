/*
 * Linear (and Identity): init, forward ranks, bias, and errors.
 *
 * - kaiming_uniform weights (and bias) in ±1/sqrt(fan_in)
 * - 2-D forward and weight.grad
 * - rank-3 applies on the last axis
 * - bias add and bias.grad; rank-1 input
 * - rank 0 and last-dim mismatch throw
 * - Identity returns the same tensor
 */

#include <cmath>
#include <stdexcept>
#include <vector>

#include "../../doctest/doctest.h"

#include "../../../src/autograd/autograd.h"
#include "../../../src/nn/layers/linear.h"

TEST_CASE("Linear kaiming_uniform weights land in ±1/sqrt(fan_in)") {
    nn::Linear<float> lin(16, 8);
    const float bound = 1.f / std::sqrt(16.f);
    for (float v : lin.weight.data())
        CHECK(std::abs(v) <= bound + 1e-5f);
    REQUIRE(lin.bias.has_value());
    for (float v : lin.bias->data())
        CHECK(std::abs(v) <= bound + 1e-5f);
}

TEST_CASE("Linear forward 2-D and weight.grad after backward") {
    nn::Linear<float> lin(3, 2, false);
    lin.weight.data() = {1.f, 0.f, 0.f, 0.f, 1.f, 0.f}; // (2, 3)
    tensor::Tensor<float> x({1, 3}, std::vector<float>{2.f, 3.f, 4.f}, true);
    auto y = lin.forward(x);
    CHECK(y.shape() == std::vector<int64_t>{1, 2});
    CHECK(y.data()[0] == doctest::Approx(2.f));
    CHECK(y.data()[1] == doctest::Approx(3.f));
    y.sum().backward();
    REQUIRE(lin.weight.grad() != nullptr);
    CHECK(lin.weight.grad()->data() == std::vector<float>{2.f, 3.f, 4.f, 2.f, 3.f, 4.f});
    CHECK(lin.parameters().size() == 1);
}

TEST_CASE("Linear forward rank-3 applies on the last axis") {
    nn::Linear<float> lin(2, 3, false);
    lin.weight.data() = {
        1.f, 0.f,
        0.f, 1.f,
        1.f, 1.f
    }; // (3, 2)
    tensor::Tensor<float> x({2, 2, 2}, std::vector<float>{1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f}, true);
    auto y = lin.forward(x);
    CHECK(y.shape() == std::vector<int64_t>{2, 2, 3});
    CHECK(y.data()[0] == doctest::Approx(1.f));
    CHECK(y.data()[1] == doctest::Approx(2.f));
    CHECK(y.data()[2] == doctest::Approx(3.f));
    CHECK(y.data()[3] == doctest::Approx(3.f));
    CHECK(y.data()[4] == doctest::Approx(4.f));
    CHECK(y.data()[5] == doctest::Approx(7.f));
    CHECK(y.data()[6] == doctest::Approx(5.f));
    CHECK(y.data()[7] == doctest::Approx(6.f));
    CHECK(y.data()[8] == doctest::Approx(11.f));
    CHECK(y.data()[9] == doctest::Approx(7.f));
    CHECK(y.data()[10] == doctest::Approx(8.f));
    CHECK(y.data()[11] == doctest::Approx(15.f));
    y.sum().backward();
    REQUIRE(lin.weight.grad() != nullptr);
}

TEST_CASE("Linear with bias adds b and fills bias.grad") {
    nn::Linear<float> lin(3, 2, true);
    lin.weight.data() = {1.f, 0.f, 0.f, 0.f, 1.f, 0.f};
    lin.bias->data() = {1.f, 2.f};
    tensor::Tensor<float> x({3}, std::vector<float>{2.f, 3.f, 4.f}, true);
    auto y = lin.forward(x);
    CHECK(y.shape() == std::vector<int64_t>{2});
    CHECK(y.data()[0] == doctest::Approx(3.f));
    CHECK(y.data()[1] == doctest::Approx(5.f));
    y.sum().backward();
    REQUIRE(lin.bias->grad() != nullptr);
    CHECK(lin.bias->grad()->data() == std::vector<float>{1.f, 1.f});
    REQUIRE(lin.weight.grad() != nullptr);
    CHECK(lin.weight.grad()->data() == std::vector<float>{2.f, 3.f, 4.f, 2.f, 3.f, 4.f});
}

TEST_CASE("Linear rejects rank 0 and last-dim mismatch") {
    nn::Linear<float> lin(3, 2);
    tensor::Tensor<float> scalar({}, std::vector<float>{1.f}, false);
    CHECK_THROWS_AS(lin.forward(scalar), std::invalid_argument);
    tensor::Tensor<float> bad({2}, std::vector<float>{1.f, 2.f}, false);
    CHECK_THROWS_AS(lin.forward(bad), std::invalid_argument);
}

TEST_CASE("Identity returns the same tensor") {
    nn::Identity<float> id;
    tensor::Tensor<float> x({3}, std::vector<float>{1.f, 2.f, 3.f}, true);
    auto y = id.forward(x);
    CHECK(y.data().data() == x.data().data());
    CHECK(id.parameters().empty());
}
