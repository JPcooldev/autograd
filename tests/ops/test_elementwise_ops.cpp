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
