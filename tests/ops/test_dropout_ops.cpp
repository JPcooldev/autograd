/*
 * Inverted dropout: eval identity, train mask, and backward.
 *
 * - eval is identity (same buffer)
 * - p=0 in train is identity
 * - same seed is deterministic
 * - survivors scaled by 1/(1-p)
 * - backward follows the saved mask
 * - p outside [0, 1) throws
 * - eval with requires_grad stays on the original leaf
 */

#include <stdexcept>
#include <vector>

#include "../doctest/doctest.h"

#include "../../src/autograd/autograd.h"
#include "../../src/ops/dropout_ops.h"

TEST_CASE("dropout eval is identity") {
    const tensor::Tensor<float> x({4}, std::vector<float>{1.f, 2.f, 3.f, 4.f}, false);
    const auto y = ops::dropout(x, 0.9, false);
    CHECK(y.data() == x.data());
    CHECK(y.data().data() == x.data().data());
}

TEST_CASE("dropout p=0 is identity in train") {
    const tensor::Tensor<float> x({4}, std::vector<float>{1.f, 2.f, 3.f, 4.f}, false);
    const auto y = ops::dropout(x, 0.0, true);
    CHECK(y.data() == x.data());
    CHECK(y.data().data() == x.data().data());
}

TEST_CASE("dropout with seed is deterministic") {
    const tensor::Tensor<float> x({8}, std::vector<float>(8, 1.f), false);
    const auto a = ops::dropout(x, 0.5, true, 7);
    const auto b = ops::dropout(x, 0.5, true, 7);
    CHECK(a.data() == b.data());
}

TEST_CASE("inverted dropout scales survivors by 1/(1-p)") {
    const tensor::Tensor<float> x({1}, std::vector<float>{2.f}, false);
    const auto y = ops::dropout(x, 0.5, true, 1);
    const float v = y.data()[0];
    CHECK((v == doctest::Approx(0.f) || v == doctest::Approx(4.f)));
}

TEST_CASE("dropout backward uses the saved mask") {
    tensor::Tensor<float> x({4}, std::vector<float>{1.f, 1.f, 1.f, 1.f}, true);
    auto y = ops::dropout(x, 0.5, true, 42);
    y.sum().backward();
    REQUIRE(x.grad() != nullptr);
    for (size_t i = 0; i < 4; ++i) {
        if (y.data()[i] == 0.f)
            CHECK(x.grad()->data()[i] == doctest::Approx(0.f));
        else
            CHECK(x.grad()->data()[i] == doctest::Approx(2.f));
    }
}

TEST_CASE("dropout p outside [0, 1) throws") {
    const tensor::Tensor<float> x({2}, std::vector<float>{1.f, 2.f}, false);
    CHECK_THROWS_AS(ops::dropout(x, -0.1, true), std::invalid_argument);
    CHECK_THROWS_AS(ops::dropout(x, 1.0, true), std::invalid_argument);
}

TEST_CASE("dropout eval with requires_grad stays on the original leaf") {
    tensor::Tensor<float> x({3}, std::vector<float>{1.f, 2.f, 3.f}, true);
    auto y = ops::dropout(x, 0.9, false);
    CHECK(y.data().data() == x.data().data());
    y.sum().backward();
    REQUIRE(x.grad() != nullptr);
    CHECK(x.grad()->data() == std::vector<float>{1.f, 1.f, 1.f});
}
