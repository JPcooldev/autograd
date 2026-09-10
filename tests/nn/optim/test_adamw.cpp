/*
 * AdamW: decoupled weight decay vs Adam's coupled λ.
 *
 * - λ is applied outside moments
 * - one step with λ=0; one step with λ>0
 * - two steps persist moments (t=2 bias correction)
 * - two steps with decoupled λ
 * - missing grad does not bump t
 * - zero_grad clears grads without wiping m/v
 * - Adam and AdamW match at λ=0 and differ at λ>0
 */

#include <vector>

#include "../../doctest/doctest.h"

#include "../../../src/autograd/autograd.h"
#include "../../../src/nn/optim/adam.h"
#include "../../../src/nn/optim/adamw.h"

TEST_CASE("AdamW applies decoupled weight_decay outside moments") {
    tensor::Tensor<float> x({1}, std::vector<float>{2.f}, true);
    x.sum().backward();

    nn::optim::AdamW<float> opt({&x}, 0.1, 0.9, 0.999, 1e-8, 0.5);
    opt.step();

    // moments on g = 1 only: m_hat = 1, v_hat = 1
    // θ ← 2 - 0.1*1/(1+1e-8) - 0.1*0.5*2 ≈ 1.8
    CHECK(x.data()[0] == doctest::Approx(1.8f));
}

TEST_CASE("AdamW step with zero weight_decay on a dummy vector parameter") {
    tensor::Tensor<float> x({2}, std::vector<float>{2.f, 4.f}, true);
    tensor::Tensor<float> g({2}, std::vector<float>{1.f, 3.f}, false);
    x.accumulate_grad(g);

    nn::optim::AdamW<float> opt({&x}, 0.1, 0.9, 0.999, 1e-8, 0.0);
    opt.step();

    // same as Adam at t = 1 with wd = 0
    CHECK(x.data()[0] == doctest::Approx(1.900000001f));
    CHECK(x.data()[1] == doctest::Approx(3.900000000333f));
}

TEST_CASE("AdamW step with weight_decay on a dummy vector parameter") {
    tensor::Tensor<float> x({2}, std::vector<float>{2.f, 4.f}, true);
    tensor::Tensor<float> g({2}, std::vector<float>{1.f, 3.f}, false);
    x.accumulate_grad(g);

    nn::optim::AdamW<float> opt({&x}, 0.1, 0.9, 0.999, 1e-8, 0.5);
    opt.step();

    // adam step ≈ lr, plus decoupled decay lr*wd*θ
    // θ ← [2, 4] - 0.1 - 0.1*0.5*[2, 4] = [1.8, 3.7]
    CHECK(x.data()[0] == doctest::Approx(1.800000001f));
    CHECK(x.data()[1] == doctest::Approx(3.700000000333f));
}

TEST_CASE("AdamW two steps persist moments and apply bias correction at t=2") {
    tensor::Tensor<float> x({1}, std::vector<float>{2.f}, true);
    nn::optim::AdamW<float> opt({&x}, 0.1, 0.9, 0.999, 1e-8, 0.0);

    tensor::Tensor<float> g1({1}, std::vector<float>{1.f}, false);
    x.accumulate_grad(g1);
    opt.step();
    CHECK(x.data()[0] == doctest::Approx(1.900000001f));

    opt.zero_grad();
    CHECK(x.grad() == nullptr);

    tensor::Tensor<float> g2({1}, std::vector<float>{2.f}, false);
    x.accumulate_grad(g2);
    opt.step();

    // same arithmetic as Adam with wd = 0: θ ≈ 1.803481799
    CHECK(x.data()[0] == doctest::Approx(1.803481799f));
}

TEST_CASE("AdamW two steps with decoupled weight_decay") {
    tensor::Tensor<float> x({1}, std::vector<float>{2.f}, true);
    nn::optim::AdamW<float> opt({&x}, 0.1, 0.9, 0.999, 1e-8, 0.5);

    tensor::Tensor<float> g1({1}, std::vector<float>{1.f}, false);
    x.accumulate_grad(g1);
    opt.step();

    opt.zero_grad();
    tensor::Tensor<float> g2({1}, std::vector<float>{2.f}, false);
    x.accumulate_grad(g2);
    opt.step();

    // t = 1: θ ≈ 1.800000001  (moments on g only, m = 0.1, v = 0.001)
    // t = 2: moments continue from those values; decay uses pre-update θ
    CHECK(x.data()[0] == doctest::Approx(1.613481799f));
}

TEST_CASE("AdamW skips missing gradients without advancing that parameter step") {
    tensor::Tensor<float> x({1}, std::vector<float>{2.f}, true);
    nn::optim::AdamW<float> opt({&x}, 0.1, 0.9, 0.999, 1e-8, 0.0);

    opt.step();
    x.sum().backward();
    opt.step();

    CHECK(x.data()[0] == doctest::Approx(1.9f));
}

TEST_CASE("AdamW zero_grad clears gradients without wiping optimizer state") {
    tensor::Tensor<float> x({1}, std::vector<float>{2.f}, true);
    nn::optim::AdamW<float> opt({&x}, 0.1, 0.9, 0.999, 1e-8, 0.0);

    tensor::Tensor<float> g1({1}, std::vector<float>{1.f}, false);
    x.accumulate_grad(g1);
    opt.step();
    REQUIRE(x.grad() != nullptr);
    opt.zero_grad();
    CHECK(x.grad() == nullptr);

    tensor::Tensor<float> g2({1}, std::vector<float>{2.f}, false);
    x.accumulate_grad(g2);
    opt.step();
    CHECK(x.data()[0] == doctest::Approx(1.803481799f));
}

TEST_CASE("Adam and AdamW match when weight_decay is zero") {
    tensor::Tensor<float> a({2}, std::vector<float>{2.f, 4.f}, true);
    tensor::Tensor<float> w({2}, std::vector<float>{2.f, 4.f}, true);
    tensor::Tensor<float> ga({2}, std::vector<float>{1.f, 3.f}, false);
    tensor::Tensor<float> gw({2}, std::vector<float>{1.f, 3.f}, false);
    a.accumulate_grad(ga);
    w.accumulate_grad(gw);

    nn::optim::Adam<float> adam({&a}, 0.1, 0.9, 0.999, 1e-8, 0.0);
    nn::optim::AdamW<float> adamw({&w}, 0.1, 0.9, 0.999, 1e-8, 0.0);
    adam.step();
    adamw.step();

    CHECK(a.data()[0] == doctest::Approx(w.data()[0]));
    CHECK(a.data()[1] == doctest::Approx(w.data()[1]));
}

TEST_CASE("Adam and AdamW differ when weight_decay is coupled vs decoupled") {
    tensor::Tensor<float> a({2}, std::vector<float>{2.f, 4.f}, true);
    tensor::Tensor<float> w({2}, std::vector<float>{2.f, 4.f}, true);
    tensor::Tensor<float> ga({2}, std::vector<float>{1.f, 3.f}, false);
    tensor::Tensor<float> gw({2}, std::vector<float>{1.f, 3.f}, false);
    a.accumulate_grad(ga);
    w.accumulate_grad(gw);

    nn::optim::Adam<float> adam({&a}, 0.1, 0.9, 0.999, 1e-8, 0.5);
    nn::optim::AdamW<float> adamw({&w}, 0.1, 0.9, 0.999, 1e-8, 0.5);
    adam.step();
    adamw.step();

    CHECK(a.data()[0] == doctest::Approx(1.9000000005f));
    CHECK(w.data()[0] == doctest::Approx(1.800000001f));
    CHECK(a.data()[0] != doctest::Approx(w.data()[0]));
    CHECK(a.data()[1] != doctest::Approx(w.data()[1]));
}
