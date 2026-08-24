#include <cmath>
#include <vector>

#include "../../doctest/doctest.h"

#include "../../../src/autograd/autograd.h"
#include "../../../src/nn/layers/convolutional.h"

TEST_CASE("Conv2d kaiming_uniform weights land in ±1/sqrt(fan_in)") {
    nn::Conv2d<float> conv(3, 4, 3);
    const float bound = 1.f / std::sqrt(3.f * 3.f * 3.f);
    for (float v : conv.weight.data())
        CHECK(std::abs(v) <= bound + 1e-5f);
    REQUIRE(conv.bias.has_value());
    for (float v : conv.bias->data())
        CHECK(std::abs(v) <= bound + 1e-5f);
}

TEST_CASE("Conv2d forward shape and weight.grad") {
    nn::Conv2d<float> conv(1, 1, 1, 1, 0, 1, false);
    conv.weight.data() = {2.f};
    tensor::Tensor<float> x({1, 1, 2, 2}, std::vector<float>{1.f, 2.f, 3.f, 4.f}, true);
    auto y = conv.forward(x);
    CHECK(y.shape() == std::vector<int64_t>{1, 1, 2, 2});
    CHECK(y.data() == std::vector<float>{2.f, 4.f, 6.f, 8.f});
    y.sum().backward();
    REQUIRE(conv.weight.grad() != nullptr);
    CHECK(conv.weight.grad()->data()[0] == doctest::Approx(10.f));
}

TEST_CASE("Conv1d and Conv3d output ranks") {
    nn::Conv1d<float> c1(2, 3, 3, 1, 1);
    tensor::Tensor<float> x1({2, 2, 5}, std::vector<float>(20, 0.1f), false);
    CHECK(c1.forward(x1).shape() == std::vector<int64_t>{2, 3, 5});

    nn::Conv3d<float> c3(1, 1, 2, 1, 0, 1, false);
    tensor::Tensor<float> x3({1, 1, 3, 3, 3}, std::vector<float>(27, 1.f), false);
    CHECK(c3.forward(x3).shape() == std::vector<int64_t>{1, 1, 2, 2, 2});
}

TEST_CASE("ConvTranspose2d output spatial size") {
    nn::ConvTranspose2d<float> conv(1, 1, 2, 1, 0, 1, false);
    tensor::Tensor<float> x({1, 1, 2, 2}, std::vector<float>(4, 1.f), false);
    const auto y = conv.forward(x);
    CHECK(y.shape() == std::vector<int64_t>{1, 1, 3, 3});
    CHECK(conv.parameters().size() == 1);
}
