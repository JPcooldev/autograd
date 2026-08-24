#include <cmath>
#include <vector>

#include "../doctest/doctest.h"

#include "../../src/autograd/autograd.h"

TYPE_TO_STRING(float);
TYPE_TO_STRING(int);

// ─── add ──────────────────────────────────────────────────────────────────

TEST_CASE_TEMPLATE("ops::add 2x2", T, float, int) {
    const tensor::Tensor<T> a({2, 2}, std::vector<T>{1, 2, 3, 4}, false);
    const tensor::Tensor<T> b({2, 2}, std::vector<T>{5, 6, 7, 8}, false);
    const auto result = ops::add(a, b);
    REQUIRE(result.shape() == std::vector<int64_t>{2, 2});
    const std::vector<T> expected{6, 8, 10, 12};
    for (size_t i = 0; i < expected.size(); ++i)
        CHECK(result.data()[i] == expected[i]);
}

// ─── neg ──────────────────────────────────────────────────────────────────

TEST_CASE_TEMPLATE("ops::neg 2x2", T, float, int) {
    const tensor::Tensor<T> a({2, 2}, std::vector<T>{1, -2, -3, 4}, false);
    const auto result = ops::neg(a);
    REQUIRE(result.shape() == std::vector<int64_t>{2, 2});
    const std::vector<T> expected{-1, 2, 3, -4};
    for (size_t i = 0; i < expected.size(); ++i)
        CHECK(result.data()[i] == expected[i]);
}

// ─── subtract ─────────────────────────────────────────────────────────────

TEST_CASE_TEMPLATE("ops::subtract 2x2", T, float, int) {
    const tensor::Tensor<T> a({2, 2}, std::vector<T>{5, 6, 7, 8}, false);
    const tensor::Tensor<T> b({2, 2}, std::vector<T>{1, 2, 3, 4}, false);
    const auto result = ops::subtract(a, b);
    REQUIRE(result.shape() == std::vector<int64_t>{2, 2});
    const std::vector<T> expected{4, 4, 4, 4};
    for (size_t i = 0; i < expected.size(); ++i)
        CHECK(result.data()[i] == expected[i]);
}

// ─── multiply ─────────────────────────────────────────────────────────────

TEST_CASE_TEMPLATE("ops::multiply 2x2", T, float, int) {
    const tensor::Tensor<T> a({2, 2}, std::vector<T>{1, 2, 3, 4}, false);
    const tensor::Tensor<T> b({2, 2}, std::vector<T>{2, 3, 4, 5}, false);
    const auto result = ops::multiply(a, b);
    REQUIRE(result.shape() == std::vector<int64_t>{2, 2});
    const std::vector<T> expected{2, 6, 12, 20};
    for (size_t i = 0; i < expected.size(); ++i)
        CHECK(result.data()[i] == expected[i]);
}

// ─── divide ───────────────────────────────────────────────────────────────

TEST_CASE_TEMPLATE("ops::divide 2x2", T, float, int) {
    const tensor::Tensor<T> a({2, 2}, std::vector<T>{2, 6, 12, 20}, false);
    const tensor::Tensor<T> b({2, 2}, std::vector<T>{2,  3,  4,  5}, false);
    const auto result = ops::divide(a, b);
    REQUIRE(result.shape() == std::vector<int64_t>{2, 2});
    const std::vector<T> expected{1, 2, 3, 4};
    for (size_t i = 0; i < expected.size(); ++i)
        CHECK(result.data()[i] == expected[i]);
}

// ─── power ────────────────────────────────────────────────────────────────

TEST_CASE_TEMPLATE("ops::power 2x2 exponent=2", T, float, int) {
    const tensor::Tensor<T> a({2, 2}, std::vector<T>{1, 2, 3, 4}, false);
    const auto result = ops::power(a, static_cast<T>(2));
    REQUIRE(result.shape() == std::vector<int64_t>{2, 2});
    const std::vector<T> expected{1, 4, 9, 16};
    for (size_t i = 0; i < expected.size(); ++i)
        CHECK(result.data()[i] == expected[i]);
}

// ─── abs ──────────────────────────────────────────────────────────────────

TEST_CASE_TEMPLATE("ops::abs 2x2", T, float, int) {
    const tensor::Tensor<T> a({2, 2}, std::vector<T>{-1, 2, -3, 4}, false);
    const auto result = ops::abs(a);
    REQUIRE(result.shape() == std::vector<int64_t>{2, 2});
    const std::vector<T> expected{1, 2, 3, 4};
    for (size_t i = 0; i < expected.size(); ++i)
        CHECK(result.data()[i] == expected[i]);
}

// ─── exp / log / sin / cos / tan  (float only) ────────────────────────────

TEST_CASE("ops::exp 2x2") {
    const std::vector<float> input{1.F, 2.F, 3.F, 4.F};
    const tensor::Tensor<float> a({2, 2}, input, false);
    const auto result = ops::exp(a);
    REQUIRE(result.shape() == std::vector<int64_t>{2, 2});
    for (size_t i = 0; i < input.size(); ++i)
        CHECK(result.data()[i] == doctest::Approx(std::exp(input[i])));
}

TEST_CASE("ops::log 2x2") {
    const std::vector<float> input{1.F, 2.F, 3.F, 4.F};
    const tensor::Tensor<float> a({2, 2}, input, false);
    const auto result = ops::log(a);
    REQUIRE(result.shape() == std::vector<int64_t>{2, 2});
    for (size_t i = 0; i < input.size(); ++i)
        CHECK(result.data()[i] == doctest::Approx(std::log(input[i])));
}

TEST_CASE("ops::sin 2x2") {
    const std::vector<float> input{0.F, 1.F, 2.F, 3.F};
    const tensor::Tensor<float> a({2, 2}, input, false);
    const auto result = ops::sin(a);
    REQUIRE(result.shape() == std::vector<int64_t>{2, 2});
    for (size_t i = 0; i < input.size(); ++i)
        CHECK(result.data()[i] == doctest::Approx(std::sin(input[i])));
}

TEST_CASE("ops::cos 2x2") {
    const std::vector<float> input{0.F, 1.F, 2.F, 3.F};
    const tensor::Tensor<float> a({2, 2}, input, false);
    const auto result = ops::cos(a);
    REQUIRE(result.shape() == std::vector<int64_t>{2, 2});
    for (size_t i = 0; i < input.size(); ++i)
        CHECK(result.data()[i] == doctest::Approx(std::cos(input[i])));
}

TEST_CASE("ops::tan 2x2") {
    const std::vector<float> input{0.F, 0.5F, 1.F, 1.5F};
    const tensor::Tensor<float> a({2, 2}, input, false);
    const auto result = ops::tan(a);
    REQUIRE(result.shape() == std::vector<int64_t>{2, 2});
    for (size_t i = 0; i < input.size(); ++i)
        CHECK(result.data()[i] == doctest::Approx(std::tan(input[i])));
}

// ─── sigmoid (float only) ─────────────────────────────────────────────────

TEST_CASE("ops::sigmoid 2x2") {
    const std::vector<float> input{-2.F, -1.F, 0.F, 1.F, 2.F, 3.F, -3.F, 0.5F};
    const tensor::Tensor<float> a({2, 4}, input, false);
    const auto result = ops::sigmoid(a);
    REQUIRE(result.shape() == std::vector<int64_t>{2, 4});
    for (size_t i = 0; i < input.size(); ++i) {
        const float expected = 1.F / (1.F + std::exp(-input[i]));
        CHECK(result.data()[i] == doctest::Approx(expected));
    }
}

TEST_CASE("ops::sigmoid output is in (0, 1)") {
    const tensor::Tensor<float> a({2, 2}, std::vector<float>{-10.F, -1.F, 1.F, 10.F}, false);
    const auto result = ops::sigmoid(a);
    for (const float v : result.data()) {
        CHECK(v > 0.F);
        CHECK(v < 1.F);
    }
}

// ─── relu (float and int) ─────────────────────────────────────────────────

TEST_CASE_TEMPLATE("ops::relu zeros negatives 2x2", T, float, int) {
    const tensor::Tensor<T> a({2, 2}, std::vector<T>{-3, -1, 0, 4}, false);
    const auto result = ops::relu(a);
    REQUIRE(result.shape() == std::vector<int64_t>{2, 2});
    const std::vector<T> expected{0, 0, 0, 4};
    for (size_t i = 0; i < expected.size(); ++i)
        CHECK(result.data()[i] == expected[i]);
}

TEST_CASE_TEMPLATE("ops::relu passes positive values unchanged", T, float, int) {
    const tensor::Tensor<T> a({2, 2}, std::vector<T>{1, 2, 3, 4}, false);
    const auto result = ops::relu(a);
    REQUIRE(result.shape() == std::vector<int64_t>{2, 2});
    for (size_t i = 0; i < result.data().size(); ++i)
        CHECK(result.data()[i] == a.data()[i]);
}

TEST_CASE("ops::sinh 2x2") {
    const tensor::Tensor<float> a({2, 2}, std::vector<float>{0.f, 1.f, -1.f, 0.5f}, false);
    const auto result = a.sinh();
    const std::vector<float> expected{
        0.f,
        std::sinh(1.f),
        std::sinh(-1.f),
        std::sinh(0.5f)
    };
    for (size_t i = 0; i < expected.size(); ++i)
        CHECK(result.data()[i] == doctest::Approx(expected[i]));
}

TEST_CASE("ops::cosh 2x2") {
    const tensor::Tensor<float> a({2}, std::vector<float>{0.f, 1.f}, false);
    const auto result = a.cosh();
    CHECK(result.data()[0] == doctest::Approx(1.f));
    CHECK(result.data()[1] == doctest::Approx(std::cosh(1.f)));
}

TEST_CASE("ops::tanh 2x2") {
    const tensor::Tensor<float> a({2}, std::vector<float>{0.f, 1.f}, false);
    const auto result = a.tanh();
    CHECK(result.data()[0] == doctest::Approx(0.f));
    CHECK(result.data()[1] == doctest::Approx(std::tanh(1.f)));
}

TEST_CASE("ops::silu is x * sigmoid(x)") {
    const tensor::Tensor<float> a({2}, std::vector<float>{0.f, 2.f}, false);
    const auto result = a.silu();
    const float sig2 = 1.f / (1.f + std::exp(-2.f));
    CHECK(result.data()[0] == doctest::Approx(0.f));
    CHECK(result.data()[1] == doctest::Approx(2.f * sig2));
}

TEST_CASE("ops::gelu tanh approximation") {
    const tensor::Tensor<float> a({2}, std::vector<float>{0.f, 1.f}, false);
    const auto result = a.gelu();
    const float k = static_cast<float>(std::sqrt(2.0 / 3.14159265358979323846));
    const float x = 1.f;
    const float expected = 0.5f * x * (1.f + std::tanh(k * (x + 0.044715f * x * x * x)));
    CHECK(result.data()[0] == doctest::Approx(0.f));
    CHECK(result.data()[1] == doctest::Approx(expected));
}

TEST_CASE("ops::sqrt 1-D") {
    const tensor::Tensor<float> a({2}, std::vector<float>{4.f, 9.f}, false);
    const auto result = ops::sqrt(a);
    CHECK(result.shape() == std::vector<int64_t>{2});
    CHECK(result.data()[0] == doctest::Approx(2.f));
    CHECK(result.data()[1] == doctest::Approx(3.f));
}

TEST_CASE("ops::sqrt backward is 0.5 / sqrt(x)") {
    tensor::Tensor<float> x({2}, std::vector<float>{4.f, 9.f}, true);
    x.sqrt().sum().backward();
    REQUIRE(x.grad() != nullptr);
    CHECK(x.grad()->data()[0] == doctest::Approx(0.25f));
    CHECK(x.grad()->data()[1] == doctest::Approx(1.f / 6.f));
}
