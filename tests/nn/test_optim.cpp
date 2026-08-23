#include <vector>

#include "../doctest/doctest.h"

#include "../../src/autograd/autograd.h"
#include "../../src/nn/optim/adam.h"
#include "../../src/nn/optim/adamw.h"
#include "../../src/nn/optim/sgd.h"

TEST_CASE("SGD weight_decay uses standard optimizer coefficient") {
    tensor::Tensor<float> x({1}, std::vector<float>{2.f}, true);
    x.sum().backward();

    nn::optim::SGD<float> opt({&x}, 0.1, 0.5);
    opt.step();

    CHECK(x.data()[0] == doctest::Approx(1.8f));
}

TEST_CASE("Adam weight_decay is coupled into moments with standard coefficient") {
    tensor::Tensor<float> x({1}, std::vector<float>{2.f}, true);
    x.sum().backward();

    nn::optim::Adam<float> opt({&x}, 0.1, 0.9, 0.999, 1e-8, 0.5);
    opt.step();

    CHECK(x.data()[0] == doctest::Approx(1.9f));
}

TEST_CASE("AdamW applies decoupled weight_decay outside moments") {
    tensor::Tensor<float> x({1}, std::vector<float>{2.f}, true);
    x.sum().backward();

    nn::optim::AdamW<float> opt({&x}, 0.1, 0.9, 0.999, 1e-8, 0.5);
    opt.step();

    CHECK(x.data()[0] == doctest::Approx(1.8f));
}

TEST_CASE("Adam skips missing gradients without advancing that parameter step") {
    tensor::Tensor<float> x({1}, std::vector<float>{2.f}, true);
    nn::optim::Adam<float> opt({&x}, 0.1);

    opt.step();
    x.sum().backward();
    opt.step();

    CHECK(x.data()[0] == doctest::Approx(1.9f));
}

TEST_CASE("AdamW skips missing gradients without advancing that parameter step") {
    tensor::Tensor<float> x({1}, std::vector<float>{2.f}, true);
    nn::optim::AdamW<float> opt({&x}, 0.1, 0.9, 0.999, 1e-8, 0.0);

    opt.step();
    x.sum().backward();
    opt.step();

    CHECK(x.data()[0] == doctest::Approx(1.9f));
}
