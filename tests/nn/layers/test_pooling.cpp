/*
 * Pooling layers: output size and a few values.
 *
 * - MaxPool2d / AvgPool2d output size
 * - GlobalAvgPool2d keeps (N, C)
 * - GlobalMaxPool2d returns the spatial max
 * - MaxPool1d values and MaxPool3d rank
 */

#include <vector>

#include "../../doctest/doctest.h"

#include "../../../src/autograd/autograd.h"
#include "../../../src/nn/layers/pooling.h"

TEST_CASE("MaxPool2d and AvgPool2d shapes") {
    tensor::Tensor<float> x({1, 1, 4, 4}, std::vector<float>(16, 1.f), false);
    nn::MaxPool2d<float> mx(2);
    nn::AvgPool2d<float> av(2);
    CHECK(mx.forward(x).shape() == std::vector<int64_t>{1, 1, 2, 2});
    CHECK(av.forward(x).shape() == std::vector<int64_t>{1, 1, 2, 2});
    CHECK(mx.parameters().empty());
}

TEST_CASE("GlobalAvgPool2d keeps (N, C)") {
    tensor::Tensor<float> x({2, 3, 4, 4}, std::vector<float>(96, 2.f), false);
    nn::GlobalAvgPool2d<float> g;
    const auto y = g.forward(x);
    CHECK(y.shape() == std::vector<int64_t>{2, 3});
    CHECK(y.data()[0] == doctest::Approx(2.f));
}

TEST_CASE("GlobalMaxPool2d") {
    tensor::Tensor<float> x({1, 1, 2, 2}, std::vector<float>{1.f, 5.f, 2.f, 3.f}, false);
    nn::GlobalMaxPool2d<float> g;
    CHECK(g.forward(x).data()[0] == doctest::Approx(5.f));
}

TEST_CASE("MaxPool1d / MaxPool3d ranks") {
    nn::MaxPool1d<float> p1(2);
    tensor::Tensor<float> x1({1, 1, 4}, std::vector<float>{1.f, 2.f, 3.f, 4.f}, false);
    CHECK(p1.forward(x1).shape() == std::vector<int64_t>{1, 1, 2});
    CHECK(p1.forward(x1).data() == std::vector<float>{2.f, 4.f});

    nn::MaxPool3d<float> p3(2);
    tensor::Tensor<float> x3({1, 1, 2, 2, 2}, std::vector<float>(8, 1.f), false);
    CHECK(p3.forward(x3).shape() == std::vector<int64_t>{1, 1, 1, 1, 1});
}
