/*
 * Max/avg/global pooling forward and backward.
 *
 * - max_pool2d / avg_pool2d on a 2×2 window
 * - max_pool1d with stride 2
 * - global_avg_pool keeps (N, C); global_max_pool
 * - max_pool2d routes grad to the argmax
 * - avg_pool2d splits 1/kvol
 * - overlapping max_pool1d accumulates at the shared argmax
 * - overlapping avg_pool1d splits 1/k onto shared cells
 * - global_max_pool backward restores the original spatial layout
 */

#include <vector>

#include "../doctest/doctest.h"

#include "../../src/autograd/autograd.h"
#include "../../src/ops/pool_ops.h"

TEST_CASE("max_pool2d kernel 2 on 2x2") {
    const tensor::Tensor<float> x({1, 1, 2, 2}, std::vector<float>{1.f, 2.f, 3.f, 4.f}, false);
    const auto y = ops::max_pool2d(x, 2);
    CHECK(y.shape() == std::vector<int64_t>{1, 1, 1, 1});
    CHECK(y.data()[0] == doctest::Approx(4.f));
}

TEST_CASE("avg_pool2d kernel 2 on 2x2") {
    const tensor::Tensor<float> x({1, 1, 2, 2}, std::vector<float>{1.f, 2.f, 3.f, 4.f}, false);
    const auto y = ops::avg_pool2d(x, 2);
    CHECK(y.shape() == std::vector<int64_t>{1, 1, 1, 1});
    CHECK(y.data()[0] == doctest::Approx(2.5f));
}

TEST_CASE("max_pool1d") {
    const tensor::Tensor<float> x({1, 1, 4}, std::vector<float>{1.f, 3.f, 2.f, 0.f}, false);
    const auto y = ops::max_pool1d(x, 2, 2);
    CHECK(y.shape() == std::vector<int64_t>{1, 1, 2});
    CHECK(y.data()[0] == doctest::Approx(3.f));
    CHECK(y.data()[1] == doctest::Approx(2.f));
}

TEST_CASE("global_avg_pool keeps (N, C)") {
    const tensor::Tensor<float> x({1, 2, 2, 2}, std::vector<float>{1.f, 1.f, 1.f, 1.f, 2.f, 2.f, 2.f, 2.f}, false);
    const auto y = ops::global_avg_pool(x);
    CHECK(y.shape() == std::vector<int64_t>{1, 2});
    CHECK(y.data()[0] == doctest::Approx(1.f));
    CHECK(y.data()[1] == doctest::Approx(2.f));
}

TEST_CASE("global_max_pool") {
    const tensor::Tensor<float> x({1, 1, 2, 2}, std::vector<float>{1.f, 4.f, 2.f, 3.f}, false);
    const auto y = ops::global_max_pool(x);
    CHECK(y.shape() == std::vector<int64_t>{1, 1});
    CHECK(y.data()[0] == doctest::Approx(4.f));
}

TEST_CASE("max_pool2d backward routes to the argmax") {
    tensor::Tensor<float> x({1, 1, 2, 2}, std::vector<float>{1.f, 2.f, 3.f, 4.f}, true);
    ops::max_pool2d(x, 2).sum().backward();
    REQUIRE(x.grad() != nullptr);
    CHECK(x.grad()->data()[0] == doctest::Approx(0.f));
    CHECK(x.grad()->data()[1] == doctest::Approx(0.f));
    CHECK(x.grad()->data()[2] == doctest::Approx(0.f));
    CHECK(x.grad()->data()[3] == doctest::Approx(1.f));
}

TEST_CASE("avg_pool2d backward splits uniformly") {
    tensor::Tensor<float> x({1, 1, 2, 2}, std::vector<float>{1.f, 2.f, 3.f, 4.f}, true);
    ops::avg_pool2d(x, 2).sum().backward();
    REQUIRE(x.grad() != nullptr);
    for (int i = 0; i < 4; ++i)
        CHECK(x.grad()->data()[static_cast<size_t>(i)] == doctest::Approx(0.25f));
}

TEST_CASE("overlapping max_pool1d accumulates at the shared argmax") {
    tensor::Tensor<float> x({1, 1, 3}, std::vector<float>{1.f, 3.f, 2.f}, true);
    ops::max_pool1d(x, 2, 1).sum().backward();
    REQUIRE(x.grad() != nullptr);
    CHECK(x.grad()->data()[0] == doctest::Approx(0.f));
    CHECK(x.grad()->data()[1] == doctest::Approx(2.f));
    CHECK(x.grad()->data()[2] == doctest::Approx(0.f));
}

TEST_CASE("overlapping avg_pool1d splits 1/k onto shared cells") {
    tensor::Tensor<float> x({1, 1, 3}, std::vector<float>{1.f, 1.f, 1.f}, true);
    ops::avg_pool1d(x, 2, 1).sum().backward();
    REQUIRE(x.grad() != nullptr);
    CHECK(x.grad()->data()[0] == doctest::Approx(0.5f));
    CHECK(x.grad()->data()[1] == doctest::Approx(1.f));
    CHECK(x.grad()->data()[2] == doctest::Approx(0.5f));
}

TEST_CASE("global_max_pool backward restores the original spatial shape") {
    tensor::Tensor<float> x({1, 1, 2, 2}, std::vector<float>{1.f, 4.f, 2.f, 3.f}, true);
    ops::global_max_pool(x).sum().backward();
    REQUIRE(x.grad() != nullptr);
    CHECK(x.grad()->shape() == x.shape());
    CHECK(x.grad()->data() == std::vector<float>{0.f, 1.f, 0.f, 0.f});
}
