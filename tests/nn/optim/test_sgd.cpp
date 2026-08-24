#include <vector>

#include "../../doctest/doctest.h"

#include "../../../src/autograd/autograd.h"
#include "../../../src/nn/optim/sgd.h"

TEST_CASE("SGD weight_decay uses standard optimizer coefficient") {
    tensor::Tensor<float> x({1}, std::vector<float>{2.f}, true);
    x.sum().backward();

    nn::optim::SGD<float> opt({&x}, 0.1, 0.5);
    opt.step();

    // g = 1,  θ ← 2 - 0.1*(1 + 0.5*2) = 1.8
    CHECK(x.data()[0] == doctest::Approx(1.8f));
}

TEST_CASE("SGD step with zero weight_decay on a 2x2 parameter") {
    tensor::Tensor<float> x({2, 2}, std::vector<float>{1.f, 2.f, 3.f, 4.f}, true);
    x.sum().backward();

    nn::optim::SGD<float> opt({&x}, 0.1, 0.0);
    opt.step();

    // g = 1 for each element,  θ ← θ - 0.1
    const std::vector<float> expected{0.9f, 1.9f, 2.9f, 3.9f};
    for (size_t i = 0; i < expected.size(); ++i)
        CHECK(x.data()[i] == doctest::Approx(expected[i]));
}

TEST_CASE("SGD step with weight_decay on a dummy vector parameter") {
    tensor::Tensor<float> x({2}, std::vector<float>{2.f, 4.f}, true);
    tensor::Tensor<float> g({2}, std::vector<float>{1.f, 3.f}, false);
    x.accumulate_grad(g);

    nn::optim::SGD<float> opt({&x}, 0.1, 0.5);
    opt.step();

    // θ ← θ - 0.1*(g + 0.5*θ)  →  [2-0.1*2, 4-0.1*5] = [1.8, 3.5]
    CHECK(x.data()[0] == doctest::Approx(1.8f));
    CHECK(x.data()[1] == doctest::Approx(3.5f));
}

TEST_CASE("SGD skips parameters without a gradient") {
    tensor::Tensor<float> a({1}, std::vector<float>{2.f}, true);
    tensor::Tensor<float> b({1}, std::vector<float>{4.f}, true);
    b.sum().backward();

    nn::optim::SGD<float> opt({&a, &b}, 0.1);
    opt.step();

    CHECK(a.data()[0] == 2.f);
    CHECK(b.data()[0] == doctest::Approx(3.9f));
}

TEST_CASE("SGD skips parameters that do not require grad") {
    tensor::Tensor<float> frozen({1}, std::vector<float>{2.f}, false);
    tensor::Tensor<float> live({1}, std::vector<float>{2.f}, true);
    live.sum().backward();

    nn::optim::SGD<float> opt({&frozen, &live}, 0.1);
    opt.step();

    CHECK(frozen.data()[0] == 2.f);
    CHECK(live.data()[0] == doctest::Approx(1.9f));
}

TEST_CASE("SGD zero_grad clears accumulated gradients") {
    tensor::Tensor<float> x({2}, std::vector<float>{1.f, 2.f}, true);
    x.sum().backward();
    REQUIRE(x.grad() != nullptr);

    nn::optim::SGD<float> opt({&x}, 0.1);
    opt.zero_grad();
    CHECK(x.grad() == nullptr);
}
