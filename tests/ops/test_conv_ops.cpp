/*
 * Conv and conv_transpose: small kernels, geometry, and a few Jacobians.
 *
 * - conv1d 1×1 scale; kernel-2 stride-1
 * - conv2d 1×1; conv3d spatial size and all-ones window
 * - conv1d + bias
 * - conv_transpose1d size and values
 * - conv1d backward into weight and input
 * - conv1d padding, stride>1, dilation
 * - conv2d backward into input and weight
 */

#include <cmath>
#include <vector>

#include "../doctest/doctest.h"

#include "../../src/autograd/autograd.h"
#include "../../src/ops/conv_ops.h"

TEST_CASE("conv1d 1x1 kernel is a per-channel scale") {
    const tensor::Tensor<float> x({1, 1, 3}, std::vector<float>{1.f, 2.f, 3.f}, false);
    const tensor::Tensor<float> w({1, 1, 1}, std::vector<float>{2.f}, false);
    const auto y = ops::conv1d(x, w);
    CHECK(y.shape() == std::vector<int64_t>{1, 1, 3});
    CHECK(y.data() == std::vector<float>{2.f, 4.f, 6.f});
}

TEST_CASE("conv1d kernel-2 stride-1") {
    const tensor::Tensor<float> x({1, 1, 3}, std::vector<float>{1.f, 2.f, 3.f}, false);
    const tensor::Tensor<float> w({1, 1, 2}, std::vector<float>{1.f, 1.f}, false);
    const auto y = ops::conv1d(x, w);
    CHECK(y.shape() == std::vector<int64_t>{1, 1, 2});
    CHECK(y.data()[0] == doctest::Approx(3.f));
    CHECK(y.data()[1] == doctest::Approx(5.f));
}

TEST_CASE("conv2d 1x1 kernel") {
    const tensor::Tensor<float> x({1, 1, 2, 2}, std::vector<float>{1.f, 2.f, 3.f, 4.f}, false);
    const tensor::Tensor<float> w({1, 1, 1, 1}, std::vector<float>{2.f}, false);
    const auto y = ops::conv2d(x, w);
    CHECK(y.shape() == std::vector<int64_t>{1, 1, 2, 2});
    CHECK(y.data() == std::vector<float>{2.f, 4.f, 6.f, 8.f});
}

TEST_CASE("conv3d output spatial size") {
    const tensor::Tensor<float> x({1, 1, 3, 3, 3}, std::vector<float>(27, 1.f), false);
    const tensor::Tensor<float> w({1, 1, 2, 2, 2}, std::vector<float>(8, 1.f), false);
    const auto y = ops::conv3d(x, w);
    CHECK(y.shape() == std::vector<int64_t>{1, 1, 2, 2, 2});
    CHECK(y.data()[0] == doctest::Approx(8.f));
}

TEST_CASE("conv1d with bias adds the bias") {
    const tensor::Tensor<float> x({1, 1, 2}, std::vector<float>{1.f, 1.f}, false);
    const tensor::Tensor<float> w({1, 1, 1}, std::vector<float>{1.f}, false);
    const tensor::Tensor<float> b({1}, std::vector<float>{0.5f}, false);
    const auto y = ops::conv1d(x, w, &b);
    CHECK(y.data()[0] == doctest::Approx(1.5f));
    CHECK(y.data()[1] == doctest::Approx(1.5f));
}

TEST_CASE("conv_transpose1d shape") {
    const tensor::Tensor<float> x({1, 1, 2}, std::vector<float>{1.f, 1.f}, false);
    const tensor::Tensor<float> w({1, 1, 2}, std::vector<float>{1.f, 1.f}, false);
    const auto y = ops::conv_transpose1d(x, w);
    CHECK(y.shape() == std::vector<int64_t>{1, 1, 3});
}

TEST_CASE("conv1d backward fills weight.grad") {
    tensor::Tensor<float> x({1, 1, 2}, std::vector<float>{1.f, 2.f}, true);
    tensor::Tensor<float> w({1, 1, 1}, std::vector<float>{3.f}, true);
    ops::conv1d(x, w).sum().backward();
    REQUIRE(w.grad() != nullptr);
    CHECK(w.grad()->data()[0] == doctest::Approx(3.f)); // 1+2
    REQUIRE(x.grad() != nullptr);
    CHECK(x.grad()->data()[0] == doctest::Approx(3.f));
    CHECK(x.grad()->data()[1] == doctest::Approx(3.f));
}

TEST_CASE("conv1d padding dilation and stride") {
    const tensor::Tensor<float> x({1, 1, 3}, std::vector<float>{1.f, 2.f, 3.f}, false);
    const tensor::Tensor<float> w({1, 1, 3}, std::vector<float>{1.f, 1.f, 1.f}, false);
    const tensor::Tensor<float>* no_bias = nullptr;
    const auto padded = ops::conv1d(x, w, no_bias, 1, 1, 1);
    CHECK(padded.shape() == std::vector<int64_t>{1, 1, 3});
    CHECK(padded.data() == std::vector<float>{3.f, 6.f, 5.f});

    const tensor::Tensor<float> xs({1, 1, 5}, std::vector<float>{1.f, 2.f, 3.f, 4.f, 5.f}, false);
    const tensor::Tensor<float> w1({1, 1, 1}, std::vector<float>{1.f}, false);
    const auto strided = ops::conv1d(xs, w1, no_bias, 2, 0, 1);
    CHECK(strided.shape() == std::vector<int64_t>{1, 1, 3});
    CHECK(strided.data() == std::vector<float>{1.f, 3.f, 5.f});

    const tensor::Tensor<float> xd({1, 1, 5}, std::vector<float>{1.f, 2.f, 3.f, 4.f, 5.f}, false);
    const tensor::Tensor<float> wd({1, 1, 2}, std::vector<float>{1.f, 1.f}, false);
    const auto dilated = ops::conv1d(xd, wd, no_bias, 1, 0, 2);
    CHECK(dilated.shape() == std::vector<int64_t>{1, 1, 3});
    CHECK(dilated.data() == std::vector<float>{4.f, 6.f, 8.f});
}

TEST_CASE("conv_transpose1d values") {
    const tensor::Tensor<float> x({1, 1, 2}, std::vector<float>{1.f, 1.f}, false);
    const tensor::Tensor<float> w({1, 1, 2}, std::vector<float>{1.f, 1.f}, false);
    const auto y = ops::conv_transpose1d(x, w);
    CHECK(y.shape() == std::vector<int64_t>{1, 1, 3});
    CHECK(y.data() == std::vector<float>{1.f, 2.f, 1.f});
}

TEST_CASE("conv2d backward into input and weight") {
    tensor::Tensor<float> x({1, 1, 2, 2}, std::vector<float>{1.f, 2.f, 3.f, 4.f}, true);
    tensor::Tensor<float> w({1, 1, 1, 1}, std::vector<float>{2.f}, true);
    ops::conv2d(x, w).sum().backward();
    REQUIRE(w.grad() != nullptr);
    CHECK(w.grad()->data()[0] == doctest::Approx(10.f));
    REQUIRE(x.grad() != nullptr);
    CHECK(x.grad()->data() == std::vector<float>{2.f, 2.f, 2.f, 2.f});
}

TEST_CASE("conv3d fills every spatial output") {
    const tensor::Tensor<float> x({1, 1, 3, 3, 3}, std::vector<float>(27, 1.f), false);
    const tensor::Tensor<float> w({1, 1, 2, 2, 2}, std::vector<float>(8, 1.f), false);
    const auto y = ops::conv3d(x, w);
    CHECK(y.shape() == std::vector<int64_t>{1, 1, 2, 2, 2});
    for (float v : y.data())
        CHECK(v == doctest::Approx(8.f));
}
