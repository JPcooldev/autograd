#include <vector>

#include "../../doctest/doctest.h"

#include "../../../src/autograd/autograd.h"
#include "../../../src/nn/layers/normalization.h"

TEST_CASE("LayerNorm affine init is ones / zeros") {
    nn::LayerNorm<float> ln(4);
    for (float v : ln.weight.data())
        CHECK(v == doctest::Approx(1.f));
    for (float v : ln.bias.data())
        CHECK(v == doctest::Approx(0.f));
}

TEST_CASE("LayerNorm last-dim mean is ~0") {
    nn::LayerNorm<float> ln(3);
    tensor::Tensor<float> x({2, 3}, std::vector<float>{1.f, 2.f, 3.f, 4.f, 5.f, 6.f}, true);
    auto y = ln.forward(x);
    CHECK(y.shape() == x.shape());
    const float m0 = (y.data()[0] + y.data()[1] + y.data()[2]) / 3.f;
    CHECK(m0 == doctest::Approx(0.f).epsilon(1e-5));
    y.sum().backward();
    REQUIRE(ln.weight.grad() != nullptr);
}

TEST_CASE("RMSNorm has no bias and ones weight") {
    nn::RMSNorm<float> rms(3);
    CHECK(rms.parameters().size() == 1);
    for (float v : rms.weight.data())
        CHECK(v == doctest::Approx(1.f));
    tensor::Tensor<float> x({1, 3}, std::vector<float>{1.f, 2.f, 2.f}, false);
    CHECK(rms.forward(x).shape() == std::vector<int64_t>{1, 3});
}

TEST_CASE("BatchNorm1d affine ones/zeros and running buffers not in parameters") {
    nn::BatchNorm1d<float> bn(2);
    for (float v : bn.weight.data())
        CHECK(v == doctest::Approx(1.f));
    for (float v : bn.bias.data())
        CHECK(v == doctest::Approx(0.f));
    CHECK(bn.running_mean.data() == std::vector<float>{0.f, 0.f});
    CHECK(bn.running_var.data() == std::vector<float>{1.f, 1.f});
    CHECK(bn.parameters().size() == 2);
}

TEST_CASE("BatchNorm1d train updates running stats; eval uses them") {
    nn::BatchNorm1d<float> bn(1);
    tensor::Tensor<float> x({4, 1}, std::vector<float>{1.f, 3.f, 5.f, 7.f}, true);
    bn.train();
    auto y = bn.forward(x);
    CHECK(y.shape() == x.shape());
    CHECK(bn.running_mean.data()[0] != doctest::Approx(0.f));
    const float rm = bn.running_mean.data()[0];
    bn.eval();
    auto y_eval = bn.forward(x);
    CHECK(y_eval.shape() == x.shape());
    CHECK(bn.running_mean.data()[0] == doctest::Approx(rm));
    y.sum().backward();
    REQUIRE(bn.weight.grad() != nullptr);
}

TEST_CASE("BatchNorm2d accepts NCHW") {
    nn::BatchNorm2d<float> bn(2);
    tensor::Tensor<float> x({2, 2, 3, 3}, std::vector<float>(36, 0.5f), false);
    CHECK(bn.forward(x).shape() == x.shape());
}
