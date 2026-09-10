/*
 * nn::Dropout: eval identity, p=0, train/eval recurse, train actually drops.
 *
 * - eval is identity
 * - p=0 in train is identity (and backward)
 * - Module train/eval recurses into Dropout
 * - p outside [0, 1) throws
 * - train with high p zeros some elements and scales survivors
 */

#include <cmath>
#include <stdexcept>
#include <vector>

#include "../../doctest/doctest.h"

#include "../../../src/autograd/autograd.h"
#include "../../../src/nn/layers/dropout.h"
#include "../../../src/nn/module.h"

TEST_CASE("Dropout eval is identity") {
    nn::Dropout<float> drop(0.9);
    drop.eval();
    tensor::Tensor<float> x({4}, std::vector<float>{1.f, 2.f, 3.f, 4.f}, false);
    const auto y = drop.forward(x);
    CHECK(y.data() == x.data());
    CHECK(drop.parameters().empty());
}

TEST_CASE("Dropout p=0 is identity in train") {
    nn::Dropout<float> drop(0.0);
    tensor::Tensor<float> x({3}, std::vector<float>{1.f, 2.f, 3.f}, true);
    auto y = drop.forward(x);
    CHECK(y.data() == x.data());
    y.sum().backward();
    REQUIRE(x.grad() != nullptr);
}

TEST_CASE("Module train/eval recurses into Dropout") {
    class M : public nn::Module<float> {
    public:
        nn::Dropout<float> drop{0.5};
        M() { register_module("drop", drop); }
    };
    M m;
    CHECK(m.drop.training());
    m.eval();
    CHECK_FALSE(m.drop.training());
    m.train();
    CHECK(m.drop.training());
}

TEST_CASE("Dropout p outside [0, 1) throws") {
    CHECK_THROWS_AS(nn::Dropout<float>(-0.1), std::invalid_argument);
    CHECK_THROWS_AS(nn::Dropout<float>(1.0), std::invalid_argument);
}

TEST_CASE("Dropout train with high p zeros some elements") {
    nn::Dropout<float> drop(0.9);
    drop.train();
    tensor::Tensor<float> x({64}, std::vector<float>(64, 1.f), false);
    const auto y = drop.forward(x);
    int zeros = 0;
    int scaled = 0;
    for (float v : y.data()) {
        if (v == 0.f)
            ++zeros;
        else if (std::abs(v - (1.f / 0.1f)) < 1e-4f)
            ++scaled;
    }
    CHECK(zeros + scaled == 64);
    CHECK(zeros > 0);
}
