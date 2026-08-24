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
