#include <cmath>
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
    CHECK(lin.parameters().size() == 1);
}
