/*
 * Loss layers: mean values and a few input grads.
 *
 * - MSELoss of equal tensors is 0
 * - MSELoss of unequal tensors (mean of squares) and input.grad
 * - L1Loss mean and input.grad
 * - CrossEntropyLoss soft labels (ln 2) and zero grad at uniform
 * - BCEWithLogitsLoss ln 2 and grad; KLDivLoss value
 * - BCELoss from probabilities
 */

#include <cmath>
#include <vector>

#include "../../doctest/doctest.h"

#include "../../../src/autograd/autograd.h"
#include "../../../src/nn/layers/loss_fn.h"

TEST_CASE("MSELoss of equal tensors is zero") {
    nn::MSELoss<float> loss;
    tensor::Tensor<float> y({3}, std::vector<float>{1.f, 2.f, 3.f}, true);
    tensor::Tensor<float> t({3}, std::vector<float>{1.f, 2.f, 3.f}, false);
    auto l = loss.forward(y, t);
    CHECK(l.numel() == 1);
    CHECK(l.data()[0] == doctest::Approx(0.f));
    CHECK(loss.parameters().empty());
}

TEST_CASE("MSELoss of unequal tensors is mean squared error") {
    nn::MSELoss<float> loss;
    tensor::Tensor<float> y({2}, std::vector<float>{1.f, 3.f}, true);
    tensor::Tensor<float> t({2}, std::vector<float>{0.f, 1.f}, false);
    auto l = loss.forward(y, t);
    CHECK(l.data()[0] == doctest::Approx(2.5f));
    l.backward();
    REQUIRE(y.grad() != nullptr);
    CHECK(y.grad()->data()[0] == doctest::Approx(1.f));
    CHECK(y.grad()->data()[1] == doctest::Approx(2.f));
}

TEST_CASE("L1Loss and backward into input") {
    nn::L1Loss<float> loss;
    tensor::Tensor<float> y({2}, std::vector<float>{1.f, 3.f}, true);
    tensor::Tensor<float> t({2}, std::vector<float>{0.f, 1.f}, false);
    auto l = loss.forward(y, t);
    CHECK(l.data()[0] == doctest::Approx(1.5f));
    l.backward();
    REQUIRE(y.grad() != nullptr);
    CHECK(y.grad()->data()[0] == doctest::Approx(0.5f));
    CHECK(y.grad()->data()[1] == doctest::Approx(0.5f));
}

TEST_CASE("CrossEntropyLoss soft labels") {
    nn::CrossEntropyLoss<float> loss;
    tensor::Tensor<float> logits({1, 2}, std::vector<float>{0.f, 0.f}, true);
    tensor::Tensor<float> target({1, 2}, std::vector<float>{0.5f, 0.5f}, false);
    auto l = loss.forward(logits, target);
    CHECK(l.numel() == 1);
    CHECK(l.data()[0] == doctest::Approx(std::log(2.f)));
    l.backward();
    REQUIRE(logits.grad() != nullptr);
    CHECK(logits.grad()->data()[0] == doctest::Approx(0.f));
    CHECK(logits.grad()->data()[1] == doctest::Approx(0.f));
}

TEST_CASE("BCEWithLogitsLoss and KLDivLoss run") {
    nn::BCEWithLogitsLoss<float> bce;
    tensor::Tensor<float> logits({2}, std::vector<float>{0.f, 0.f}, true);
    tensor::Tensor<float> t({2}, std::vector<float>{0.f, 1.f}, false);
    auto l = bce.forward(logits, t);
    CHECK(l.numel() == 1);
    CHECK(l.data()[0] == doctest::Approx(std::log(2.f)));
    l.backward();
    REQUIRE(logits.grad() != nullptr);
    CHECK(logits.grad()->data()[0] == doctest::Approx(0.25f));
    CHECK(logits.grad()->data()[1] == doctest::Approx(-0.25f));

    nn::KLDivLoss<float> kl;
    tensor::Tensor<float> logp({2}, std::vector<float>{-1.f, -2.f}, true);
    tensor::Tensor<float> p({2}, std::vector<float>{0.5f, 0.5f}, false);
    auto kl_loss = kl.forward(logp, p);
    const float log_t = std::log(0.5f);
    CHECK(kl_loss.data()[0] == doctest::Approx(
        0.5f * (0.5f * (log_t + 1.f) + 0.5f * (log_t + 2.f))));
}

TEST_CASE("BCELoss from probabilities") {
    nn::BCELoss<float> bce;
    tensor::Tensor<float> p({2}, std::vector<float>{0.25f, 0.75f}, true);
    tensor::Tensor<float> t({2}, std::vector<float>{0.f, 1.f}, false);
    auto l = bce.forward(p, t);
    CHECK(l.data()[0] == doctest::Approx(-std::log(0.75f)));
}
