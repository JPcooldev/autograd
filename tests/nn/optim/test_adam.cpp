#include <vector>

#include "../../doctest/doctest.h"

#include "../../../src/autograd/autograd.h"
#include "../../../src/nn/optim/adam.h"

TEST_CASE("Adam weight_decay is coupled into moments with standard coefficient") {
    tensor::Tensor<float> x({1}, std::vector<float>{2.f}, true);
    x.sum().backward();

    nn::optim::Adam<float> opt({&x}, 0.1, 0.9, 0.999, 1e-8, 0.5);
    opt.step();

    // g_eff = 1 + 0.5*2 = 2
    // m = 0.2, v = 0.004, m_hat = 2, v_hat = 4
    // θ ← 2 - 0.1 * 2 / (2 + 1e-8) ≈ 1.9
    CHECK(x.data()[0] == doctest::Approx(1.9f));
}

TEST_CASE("Adam step with zero weight_decay on a dummy vector parameter") {
    tensor::Tensor<float> x({2}, std::vector<float>{2.f, 4.f}, true);
    tensor::Tensor<float> g({2}, std::vector<float>{1.f, 3.f}, false);
    x.accumulate_grad(g);

    nn::optim::Adam<float> opt({&x}, 0.1, 0.9, 0.999, 1e-8, 0.0);
    opt.step();

    // t = 1: m_hat = g, v_hat = g^2, so each element steps by ≈ lr
    // θ ← [2, 4] - 0.1 * g/|g|  →  [1.900000001, 3.900000000333]
    CHECK(x.data()[0] == doctest::Approx(1.900000001f));
    CHECK(x.data()[1] == doctest::Approx(3.900000000333f));
}

TEST_CASE("Adam step with weight_decay on a dummy vector parameter") {
    tensor::Tensor<float> x({2}, std::vector<float>{2.f, 4.f}, true);
    tensor::Tensor<float> g({2}, std::vector<float>{1.f, 3.f}, false);
    x.accumulate_grad(g);

    nn::optim::Adam<float> opt({&x}, 0.1, 0.9, 0.999, 1e-8, 0.5);
    opt.step();

    // g_eff = [1+1, 3+2] = [2, 5]; at t = 1 the Adam step is still ≈ lr
    CHECK(x.data()[0] == doctest::Approx(1.9000000005f));
    CHECK(x.data()[1] == doctest::Approx(3.9000000002f));
}

TEST_CASE("Adam two steps persist moments and apply bias correction at t=2") {
    tensor::Tensor<float> x({1}, std::vector<float>{2.f}, true);
    nn::optim::Adam<float> opt({&x}, 0.1, 0.9, 0.999, 1e-8, 0.0);

    tensor::Tensor<float> g1({1}, std::vector<float>{1.f}, false);
    x.accumulate_grad(g1);
    opt.step();
    CHECK(x.data()[0] == doctest::Approx(1.900000001f));

    opt.zero_grad();
    CHECK(x.grad() == nullptr);

    tensor::Tensor<float> g2({1}, std::vector<float>{2.f}, false);
    x.accumulate_grad(g2);
    opt.step();

    // t = 2, g = 2, moments carried from t = 1 (m = 0.1, v = 0.001):
    // m = 0.9*0.1 + 0.1*2 = 0.29
    // v = 0.999*0.001 + 0.001*4 = 0.004999
    // 1-β1^2 = 0.19,  1-β2^2 = 0.001999
    // m_hat = 0.29/0.19,  v_hat = 0.004999/0.001999
    // θ ← 1.900000001 - 0.1 * m_hat / (sqrt(v_hat) + 1e-8) ≈ 1.803481799
    CHECK(x.data()[0] == doctest::Approx(1.803481799f));
}

TEST_CASE("Adam two steps with weight_decay persist coupled moments") {
    tensor::Tensor<float> x({1}, std::vector<float>{2.f}, true);
    nn::optim::Adam<float> opt({&x}, 0.1, 0.9, 0.999, 1e-8, 0.5);

    tensor::Tensor<float> g1({1}, std::vector<float>{1.f}, false);
    x.accumulate_grad(g1);
    opt.step();

    opt.zero_grad();
    tensor::Tensor<float> g2({1}, std::vector<float>{2.f}, false);
    x.accumulate_grad(g2);
    opt.step();

    // t = 1: g_eff = 2, θ ≈ 1.9000000005
    // t = 2: g_eff = 2 + 0.5*1.9000000005, moments continue from (m = 0.2, v = 0.004)
    CHECK(x.data()[0] == doctest::Approx(1.800809475f));
}

TEST_CASE("Adam skips missing gradients without advancing that parameter step") {
    tensor::Tensor<float> x({1}, std::vector<float>{2.f}, true);
    nn::optim::Adam<float> opt({&x}, 0.1);

    opt.step();
    x.sum().backward();
    opt.step();

    // skipped step did not bump t, so this is still the t = 1 update ≈ 1.9
    CHECK(x.data()[0] == doctest::Approx(1.9f));
}

TEST_CASE("Adam zero_grad clears gradients without wiping optimizer state") {
    tensor::Tensor<float> x({1}, std::vector<float>{2.f}, true);
    nn::optim::Adam<float> opt({&x}, 0.1, 0.9, 0.999, 1e-8, 0.0);

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
