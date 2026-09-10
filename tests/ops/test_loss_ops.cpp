/*
 * Loss ops: L1/L2 and the classification / probabilistic losses.
 *
 * - l1/l2 mean and sum
 * - L1/L2 backward (1/N vs unscaled; 2/N vs 2)
 * - mse_loss aliases l2_loss
 * - nll_loss 2-D dim=-1 and backward
 * - cross_entropy fused softmax+NLL
 * - bce from probabilities; bce_with_logits at zero logits
 * - kl_div mean of t*(log t - input)
 * - shape mismatch throws
 */

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include "../doctest/doctest.h"

#include "../../src/autograd/autograd.h"
#include "../../src/ops/loss_ops.h"

// input = [1, 3], target = [0, 1]
// |diff| = [1, 2]   sum=3  mean=1.5
//  diff² = [1, 4]   sum=5  mean=2.5

TEST_CASE("l1_loss mean reduction (default)") {
    const tensor::Tensor<float> input({2}, std::vector<float>{1.f, 3.f}, false);
    const tensor::Tensor<float> target({2}, std::vector<float>{0.f, 1.f}, false);
    const auto loss = ops::l1_loss(input, target);
    CHECK(loss.shape() == std::vector<int64_t>{});
    CHECK(loss.data()[0] == doctest::Approx(1.5f));
}

TEST_CASE("l1_loss sum reduction") {
    const tensor::Tensor<float> input({2}, std::vector<float>{1.f, 3.f}, false);
    const tensor::Tensor<float> target({2}, std::vector<float>{0.f, 1.f}, false);
    const auto loss = ops::l1_loss(input, target, false);
    CHECK(loss.shape() == std::vector<int64_t>{});
    CHECK(loss.data()[0] == doctest::Approx(3.f));
}

TEST_CASE("l2_loss mean reduction (default)") {
    const tensor::Tensor<float> input({2}, std::vector<float>{1.f, 3.f}, false);
    const tensor::Tensor<float> target({2}, std::vector<float>{0.f, 1.f}, false);
    const auto loss = ops::l2_loss(input, target);
    CHECK(loss.shape() == std::vector<int64_t>{});
    CHECK(loss.data()[0] == doctest::Approx(2.5f));
}

TEST_CASE("l2_loss sum reduction") {
    const tensor::Tensor<float> input({2}, std::vector<float>{1.f, 3.f}, false);
    const tensor::Tensor<float> target({2}, std::vector<float>{0.f, 1.f}, false);
    const auto loss = ops::l2_loss(input, target, false);
    CHECK(loss.shape() == std::vector<int64_t>{});
    CHECK(loss.data()[0] == doctest::Approx(5.f));
}

TEST_CASE("L1LossBackward mean scales by 1/N") {
    tensor::Tensor<float> input({2}, std::vector<float>{1.f, 3.f}, true);
    tensor::Tensor<float> target({2}, std::vector<float>{0.f, 1.f}, true);
    auto loss = ops::l1_loss(input, target);
    loss.backward();

    REQUIRE(input.grad() != nullptr);
    REQUIRE(target.grad() != nullptr);
    CHECK(input.grad()->data()[0] == doctest::Approx(0.5f));
    CHECK(input.grad()->data()[1] == doctest::Approx(0.5f));
    CHECK(target.grad()->data()[0] == doctest::Approx(-0.5f));
    CHECK(target.grad()->data()[1] == doctest::Approx(-0.5f));
}

TEST_CASE("L1LossBackward sum does not scale by N") {
    tensor::Tensor<float> input({2}, std::vector<float>{1.f, 3.f}, true);
    tensor::Tensor<float> target({2}, std::vector<float>{0.f, 1.f}, true);
    auto loss = ops::l1_loss(input, target, false);
    loss.backward();

    REQUIRE(input.grad() != nullptr);
    REQUIRE(target.grad() != nullptr);
    CHECK(input.grad()->data()[0] == doctest::Approx(1.f));
    CHECK(input.grad()->data()[1] == doctest::Approx(1.f));
    CHECK(target.grad()->data()[0] == doctest::Approx(-1.f));
    CHECK(target.grad()->data()[1] == doctest::Approx(-1.f));
}

TEST_CASE("L2LossBackward mean scales by 2/N") {
    tensor::Tensor<float> input({2}, std::vector<float>{1.f, 3.f}, true);
    tensor::Tensor<float> target({2}, std::vector<float>{0.f, 1.f}, true);
    auto loss = ops::l2_loss(input, target);
    loss.backward();

    REQUIRE(input.grad() != nullptr);
    REQUIRE(target.grad() != nullptr);
    // 2 * [1, 2] / 2 = [1, 2]
    CHECK(input.grad()->data()[0] == doctest::Approx(1.f));
    CHECK(input.grad()->data()[1] == doctest::Approx(2.f));
    CHECK(target.grad()->data()[0] == doctest::Approx(-1.f));
    CHECK(target.grad()->data()[1] == doctest::Approx(-2.f));
}

TEST_CASE("L2LossBackward sum scales by 2") {
    tensor::Tensor<float> input({2}, std::vector<float>{1.f, 3.f}, true);
    tensor::Tensor<float> target({2}, std::vector<float>{0.f, 1.f}, true);
    auto loss = ops::l2_loss(input, target, false);
    loss.backward();

    REQUIRE(input.grad() != nullptr);
    REQUIRE(target.grad() != nullptr);
    // 2 * [1, 2] = [2, 4]
    CHECK(input.grad()->data()[0] == doctest::Approx(2.f));
    CHECK(input.grad()->data()[1] == doctest::Approx(4.f));
    CHECK(target.grad()->data()[0] == doctest::Approx(-2.f));
    CHECK(target.grad()->data()[1] == doctest::Approx(-4.f));
}

TEST_CASE("mse_loss is an alias of l2_loss") {
    const tensor::Tensor<float> input({2}, std::vector<float>{1.f, 3.f}, false);
    const tensor::Tensor<float> target({2}, std::vector<float>{0.f, 1.f}, false);
    CHECK(ops::mse_loss(input, target).data()[0] == doctest::Approx(2.5f));
    CHECK(ops::mse_loss(input, target, false).data()[0] == doctest::Approx(5.f));
}

TEST_CASE("nll_loss 2-D dim=-1 and backward") {
    tensor::Tensor<float> logp(
        {2, 2}, std::vector<float>{-1.f, -2.f, -3.f, -4.f}, true);
    tensor::Tensor<float> target(
        {2, 2}, std::vector<float>{1.f, 0.f, 0.f, 1.f}, true);
    auto loss = ops::nll_loss(logp, target);
    // n_batch = 2;  -(-1 + 0 + 0 - 4) / 2 = 2.5
    CHECK(loss.data()[0] == doctest::Approx(2.5f));
    loss.backward();
    REQUIRE(logp.grad() != nullptr);
    REQUIRE(target.grad() != nullptr);
    CHECK(logp.grad()->data()[0] == doctest::Approx(-0.5f));
    CHECK(logp.grad()->data()[1] == doctest::Approx(0.f));
    CHECK(logp.grad()->data()[2] == doctest::Approx(0.f));
    CHECK(logp.grad()->data()[3] == doctest::Approx(-0.5f));
}

TEST_CASE("cross_entropy_loss fused softmax + NLL") {
    tensor::Tensor<float> logits({1, 2}, std::vector<float>{0.f, 1.f}, true);
    tensor::Tensor<float> target({1, 2}, std::vector<float>{1.f, 0.f}, false);
    auto loss = ops::cross_entropy_loss(logits, target);
    const float e = std::exp(1.f);
    CHECK(loss.data()[0] == doctest::Approx(std::log(1.f + e)));
    loss.backward();
    REQUIRE(logits.grad() != nullptr);
    CHECK(logits.grad()->data()[0] == doctest::Approx(-e / (1.f + e)));
    CHECK(logits.grad()->data()[1] == doctest::Approx(e / (1.f + e)));
}

TEST_CASE("bce_loss forward and input.grad") {
    tensor::Tensor<float> p({2}, std::vector<float>{0.25f, 0.75f}, true);
    tensor::Tensor<float> t({2}, std::vector<float>{0.f, 1.f}, false);
    auto loss = ops::bce_loss(p, t);
    CHECK(loss.data()[0] == doctest::Approx(-std::log(0.75f)));
    loss.backward();
    REQUIRE(p.grad() != nullptr);
    const float eps = std::numeric_limits<float>::epsilon();
    const float p0 = std::max(std::min(0.25f, 1.f - eps), eps);
    const float p1 = std::max(std::min(0.75f, 1.f - eps), eps);
    CHECK(p.grad()->data()[0] == doctest::Approx((p0 - 0.f) / (p0 * (1.f - p0) * 2.f)));
    CHECK(p.grad()->data()[1] == doctest::Approx((p1 - 1.f) / (p1 * (1.f - p1) * 2.f)));
}

TEST_CASE("bce_with_logits_loss at zero logits is ln(2)") {
    tensor::Tensor<float> logits({2}, std::vector<float>{0.f, 0.f}, true);
    tensor::Tensor<float> t({2}, std::vector<float>{0.f, 1.f}, false);
    auto loss = ops::bce_with_logits_loss(logits, t);
    CHECK(loss.data()[0] == doctest::Approx(std::log(2.f)));
    loss.backward();
    REQUIRE(logits.grad() != nullptr);
    CHECK(logits.grad()->data()[0] == doctest::Approx(0.25f));
    CHECK(logits.grad()->data()[1] == doctest::Approx(-0.25f));
}

TEST_CASE("kl_div_loss mean of t * (log(t) - input)") {
    tensor::Tensor<float> logp({2}, std::vector<float>{-1.f, -2.f}, true);
    tensor::Tensor<float> t({2}, std::vector<float>{0.5f, 0.5f}, false);
    auto loss = ops::kl_div_loss(logp, t);
    const float log_t = std::log(0.5f);
    const float expected = 0.5f * (0.5f * (log_t + 1.f) + 0.5f * (log_t + 2.f));
    CHECK(loss.data()[0] == doctest::Approx(expected));
    loss.backward();
    REQUIRE(logp.grad() != nullptr);
    CHECK(logp.grad()->data()[0] == doctest::Approx(-0.25f));
    CHECK(logp.grad()->data()[1] == doctest::Approx(-0.25f));
}

TEST_CASE("loss ops throw on shape mismatch") {
    const tensor::Tensor<float> a({2}, std::vector<float>{1.f, 2.f}, false);
    const tensor::Tensor<float> b({3}, std::vector<float>{1.f, 2.f, 3.f}, false);
    CHECK_THROWS_AS(ops::l1_loss(a, b), std::invalid_argument);
    CHECK_THROWS_AS(ops::cross_entropy_loss(a, b), std::invalid_argument);
    CHECK_THROWS_AS(ops::bce_loss(a, b), std::invalid_argument);
}
