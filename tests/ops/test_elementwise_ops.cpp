/*
 * Elementwise ops: arithmetic, activations, and a few Jacobians.
 *
 * - add, neg, subtract, multiply, divide, power (float/double/int)
 * - abs, sqrt; sqrt backward
 * - exp, log, sin, cos, tan (float/double)
 * - sinh, cosh, tanh
 * - sigmoid values and range
 * - relu zeros negatives; positives unchanged
 * - silu = x * sigmoid(x); gelu tanh approx
 * - pack a transpose before the dense loop; relu-after-transpose backward
 * - divide backward; shape mismatch; divide-by-zero and log of non-positive
 * - abs/relu at 0; log and sin backward; silu backward
 */

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "../doctest/doctest.h"

#include "../../src/autograd/autograd.h"

TYPE_TO_STRING(float);
TYPE_TO_STRING(double);
TYPE_TO_STRING(int32_t);
TYPE_TO_STRING(int64_t);

// Arithmetic / abs / relu: all four tensor dtypes.
// Transcendental and smooth activations: float and double only
// (integer results would truncate to 0/1 and not exercise the op).

// ─── add ───

TEST_CASE_TEMPLATE("ops::add 2x2", T, float, double, int32_t, int64_t) {
    const tensor::Tensor<T> a({2, 2}, std::vector<T>{1, 2, 3, 4}, false);
    const tensor::Tensor<T> b({2, 2}, std::vector<T>{5, 6, 7, 8}, false);
    const auto result = ops::add(a, b);
    REQUIRE(result.shape() == std::vector<int64_t>{2, 2});
    const std::vector<T> expected{6, 8, 10, 12};
    for (size_t i = 0; i < expected.size(); ++i)
        CHECK(result.data()[i] == expected[i]);
}

// ─── neg ───

TEST_CASE_TEMPLATE("ops::neg 2x2", T, float, double, int32_t, int64_t) {
    const tensor::Tensor<T> a({2, 2}, std::vector<T>{1, -2, -3, 4}, false);
    const auto result = ops::neg(a);
    REQUIRE(result.shape() == std::vector<int64_t>{2, 2});
    const std::vector<T> expected{-1, 2, 3, -4};
    for (size_t i = 0; i < expected.size(); ++i)
        CHECK(result.data()[i] == expected[i]);
}

// ─── subtract ───

TEST_CASE_TEMPLATE("ops::subtract 2x2", T, float, double, int32_t, int64_t) {
    const tensor::Tensor<T> a({2, 2}, std::vector<T>{5, 6, 7, 8}, false);
    const tensor::Tensor<T> b({2, 2}, std::vector<T>{1, 2, 3, 4}, false);
    const auto result = ops::subtract(a, b);
    REQUIRE(result.shape() == std::vector<int64_t>{2, 2});
    const std::vector<T> expected{4, 4, 4, 4};
    for (size_t i = 0; i < expected.size(); ++i)
        CHECK(result.data()[i] == expected[i]);
}

// ─── multiply ───

TEST_CASE_TEMPLATE("ops::multiply 2x2", T, float, double, int32_t, int64_t) {
    const tensor::Tensor<T> a({2, 2}, std::vector<T>{1, 2, 3, 4}, false);
    const tensor::Tensor<T> b({2, 2}, std::vector<T>{2, 3, 4, 5}, false);
    const auto result = ops::multiply(a, b);
    REQUIRE(result.shape() == std::vector<int64_t>{2, 2});
    const std::vector<T> expected{2, 6, 12, 20};
    for (size_t i = 0; i < expected.size(); ++i)
        CHECK(result.data()[i] == expected[i]);
}

// ─── divide ───

TEST_CASE_TEMPLATE("ops::divide 2x2", T, float, double, int32_t, int64_t) {
    const tensor::Tensor<T> a({2, 2}, std::vector<T>{2, 6, 12, 20}, false);
    const tensor::Tensor<T> b({2, 2}, std::vector<T>{2,  3,  4,  5}, false);
    const auto result = ops::divide(a, b);
    REQUIRE(result.shape() == std::vector<int64_t>{2, 2});
    const std::vector<T> expected{1, 2, 3, 4};
    for (size_t i = 0; i < expected.size(); ++i)
        CHECK(result.data()[i] == expected[i]);
}

// ─── power ───

TEST_CASE_TEMPLATE("ops::power 2x2 exponent=2", T, float, double, int32_t, int64_t) {
    const tensor::Tensor<T> a({2, 2}, std::vector<T>{1, 2, 3, 4}, false);
    const auto result = ops::power(a, static_cast<T>(2));
    REQUIRE(result.shape() == std::vector<int64_t>{2, 2});
    const std::vector<T> expected{1, 4, 9, 16};
    for (size_t i = 0; i < expected.size(); ++i)
        CHECK(result.data()[i] == expected[i]);
}

// ─── abs ───

TEST_CASE_TEMPLATE("ops::abs 2x2", T, float, double, int32_t, int64_t) {
    const tensor::Tensor<T> a({2, 2}, std::vector<T>{-1, 2, -3, 4}, false);
    const auto result = ops::abs(a);
    REQUIRE(result.shape() == std::vector<int64_t>{2, 2});
    const std::vector<T> expected{1, 2, 3, 4};
    for (size_t i = 0; i < expected.size(); ++i)
        CHECK(result.data()[i] == expected[i]);
}

// ─── sqrt ───

TEST_CASE_TEMPLATE("ops::sqrt perfect squares", T, float, double, int32_t, int64_t) {
    const tensor::Tensor<T> a({2}, std::vector<T>{4, 9}, false);
    const auto result = ops::sqrt(a);
    CHECK(result.shape() == std::vector<int64_t>{2});
    CHECK(result.data()[0] == static_cast<T>(2));
    CHECK(result.data()[1] == static_cast<T>(3));
}

TEST_CASE_TEMPLATE("ops::sqrt backward is 0.5 / sqrt(x)", T, float, double) {
    tensor::Tensor<T> x({2}, std::vector<T>{4, 9}, true);
    x.sqrt().sum().backward();
    REQUIRE(x.grad() != nullptr);
    CHECK(x.grad()->data()[0] == doctest::Approx(static_cast<T>(0.25)));
    CHECK(x.grad()->data()[1] == doctest::Approx(static_cast<T>(1) / static_cast<T>(6)));
}

// ─── exp / log / sin / cos / tan ───

TEST_CASE_TEMPLATE("ops::exp 2x2", T, float, double) {
    const std::vector<T> input{1, 2, 3, 4};
    const tensor::Tensor<T> a({2, 2}, input, false);
    const auto result = ops::exp(a);
    REQUIRE(result.shape() == std::vector<int64_t>{2, 2});
    for (size_t i = 0; i < input.size(); ++i)
        CHECK(result.data()[i] == doctest::Approx(std::exp(input[i])));
}

TEST_CASE_TEMPLATE("ops::log 2x2", T, float, double) {
    const std::vector<T> input{1, 2, 3, 4};
    const tensor::Tensor<T> a({2, 2}, input, false);
    const auto result = ops::log(a);
    REQUIRE(result.shape() == std::vector<int64_t>{2, 2});
    for (size_t i = 0; i < input.size(); ++i)
        CHECK(result.data()[i] == doctest::Approx(std::log(input[i])));
}

TEST_CASE_TEMPLATE("ops::sin 2x2", T, float, double) {
    const std::vector<T> input{0, 1, 2, 3};
    const tensor::Tensor<T> a({2, 2}, input, false);
    const auto result = ops::sin(a);
    REQUIRE(result.shape() == std::vector<int64_t>{2, 2});
    for (size_t i = 0; i < input.size(); ++i)
        CHECK(result.data()[i] == doctest::Approx(std::sin(input[i])));
}

TEST_CASE_TEMPLATE("ops::cos 2x2", T, float, double) {
    const std::vector<T> input{0, 1, 2, 3};
    const tensor::Tensor<T> a({2, 2}, input, false);
    const auto result = ops::cos(a);
    REQUIRE(result.shape() == std::vector<int64_t>{2, 2});
    for (size_t i = 0; i < input.size(); ++i)
        CHECK(result.data()[i] == doctest::Approx(std::cos(input[i])));
}

TEST_CASE_TEMPLATE("ops::tan 2x2", T, float, double) {
    const std::vector<T> input{
        static_cast<T>(0),
        static_cast<T>(0.5),
        static_cast<T>(1),
        static_cast<T>(1.5)
    };
    const tensor::Tensor<T> a({2, 2}, input, false);
    const auto result = ops::tan(a);
    REQUIRE(result.shape() == std::vector<int64_t>{2, 2});
    for (size_t i = 0; i < input.size(); ++i)
        CHECK(result.data()[i] == doctest::Approx(std::tan(input[i])));
}

// ─── hyperbolic ───

TEST_CASE_TEMPLATE("ops::sinh 2x2", T, float, double) {
    const std::vector<T> input{
        static_cast<T>(0),
        static_cast<T>(1),
        static_cast<T>(-1),
        static_cast<T>(0.5)
    };
    const tensor::Tensor<T> a({2, 2}, input, false);
    const auto result = ops::sinh(a);
    for (size_t i = 0; i < input.size(); ++i)
        CHECK(result.data()[i] == doctest::Approx(std::sinh(input[i])));
}

TEST_CASE_TEMPLATE("ops::cosh", T, float, double) {
    const tensor::Tensor<T> a({2}, std::vector<T>{0, 1}, false);
    const auto result = ops::cosh(a);
    CHECK(result.data()[0] == doctest::Approx(static_cast<T>(1)));
    CHECK(result.data()[1] == doctest::Approx(std::cosh(static_cast<T>(1))));
}

TEST_CASE_TEMPLATE("ops::tanh", T, float, double) {
    const tensor::Tensor<T> a({2}, std::vector<T>{0, 1}, false);
    const auto result = ops::tanh(a);
    CHECK(result.data()[0] == doctest::Approx(static_cast<T>(0)));
    CHECK(result.data()[1] == doctest::Approx(std::tanh(static_cast<T>(1))));
}

// ─── sigmoid ───

TEST_CASE_TEMPLATE("ops::sigmoid 2x2", T, float, double) {
    const std::vector<T> input{
        static_cast<T>(-2),
        static_cast<T>(-1),
        static_cast<T>(0),
        static_cast<T>(1)
    };
    const tensor::Tensor<T> a({2, 2}, input, false);
    const auto result = ops::sigmoid(a);
    REQUIRE(result.shape() == std::vector<int64_t>{2, 2});
    for (size_t i = 0; i < input.size(); ++i) {
        const T expected = static_cast<T>(1) / (static_cast<T>(1) + static_cast<T>(std::exp(-input[i])));
        CHECK(result.data()[i] == doctest::Approx(expected));
    }
}

TEST_CASE_TEMPLATE("ops::sigmoid output is in (0, 1)", T, float, double) {
    const tensor::Tensor<T> a({2, 2}, std::vector<T>{-10, -1, 1, 10}, false);
    const auto result = ops::sigmoid(a);
    for (const T v : result.data()) {
        CHECK(v > static_cast<T>(0));
        CHECK(v < static_cast<T>(1));
    }
}

// ─── relu ───

TEST_CASE_TEMPLATE("ops::relu zeros negatives 2x2", T, float, double, int32_t, int64_t) {
    const tensor::Tensor<T> a({2, 2}, std::vector<T>{-3, -1, 0, 4}, false);
    const auto result = ops::relu(a);
    REQUIRE(result.shape() == std::vector<int64_t>{2, 2});
    const std::vector<T> expected{0, 0, 0, 4};
    for (size_t i = 0; i < expected.size(); ++i)
        CHECK(result.data()[i] == expected[i]);
}

TEST_CASE_TEMPLATE("ops::relu passes positive values unchanged", T, float, double, int32_t, int64_t) {
    const tensor::Tensor<T> a({2, 2}, std::vector<T>{1, 2, 3, 4}, false);
    const auto result = ops::relu(a);
    REQUIRE(result.shape() == std::vector<int64_t>{2, 2});
    for (size_t i = 0; i < result.data().size(); ++i)
        CHECK(result.data()[i] == a.data()[i]);
}

// ─── silu / gelu ───

TEST_CASE_TEMPLATE("ops::silu is x * sigmoid(x)", T, float, double) {
    const tensor::Tensor<T> a({2}, std::vector<T>{0, 2}, false);
    const auto result = ops::silu(a);
    const T sig2 = static_cast<T>(1) / (static_cast<T>(1) + static_cast<T>(std::exp(static_cast<T>(-2))));
    CHECK(result.data()[0] == doctest::Approx(static_cast<T>(0)));
    CHECK(result.data()[1] == doctest::Approx(static_cast<T>(2) * sig2));
}

TEST_CASE_TEMPLATE("ops::gelu tanh approximation", T, float, double) {
    const tensor::Tensor<T> a({2}, std::vector<T>{0, 1}, false);
    const auto result = ops::gelu(a);
    const T k = static_cast<T>(std::sqrt(2.0 / 3.14159265358979323846));
    const T x = static_cast<T>(1);
    const T expected = static_cast<T>(0.5) * x
        * (static_cast<T>(1) + static_cast<T>(std::tanh(k * (x + static_cast<T>(0.044715) * x * x * x))));
    CHECK(result.data()[0] == doctest::Approx(static_cast<T>(0)));
    CHECK(result.data()[1] == doctest::Approx(expected));
}

TEST_CASE("elementwise ops pack a transpose before the dense loop") {
    const tensor::Tensor<float> t({2, 3}, std::vector<float>{1.f, -2.f, 3.f, 4.f, -5.f, 6.f}, false);
    const auto r = t.transpose().relu();
    CHECK(r.shape() == std::vector<int64_t>{3, 2});
    CHECK(r.is_contiguous());
    CHECK(r.data() == std::vector<float>{1.f, 4.f, 0.f, 0.f, 3.f, 6.f});
}

TEST_CASE("relu after transpose backprops in logical order") {
    tensor::Tensor<float> x({2, 3}, std::vector<float>{1.f, -2.f, 3.f, 4.f, -5.f, 6.f}, true);
    x.transpose().relu().sum().backward();
    REQUIRE(x.grad() != nullptr);
    const std::vector<float> expected{1.f, 0.f, 1.f, 1.f, 0.f, 1.f};
    for (size_t i = 0; i < expected.size(); ++i)
        CHECK(x.grad()->data()[i] == doctest::Approx(expected[i]));
}

TEST_CASE("divide backward is g/y and -x/y^2 * g") {
    tensor::Tensor<float> x({2}, std::vector<float>{6.f, 8.f}, true);
    tensor::Tensor<float> y({2}, std::vector<float>{2.f, 4.f}, true);
    x.divide(y).sum().backward();
    REQUIRE(x.grad() != nullptr);
    REQUIRE(y.grad() != nullptr);
    CHECK(x.grad()->data()[0] == doctest::Approx(0.5f));
    CHECK(x.grad()->data()[1] == doctest::Approx(0.25f));
    CHECK(y.grad()->data()[0] == doctest::Approx(-1.5f));
    CHECK(y.grad()->data()[1] == doctest::Approx(-0.5f));
}

TEST_CASE("elementwise binary ops throw on shape mismatch") {
    const tensor::Tensor<float> a({2}, std::vector<float>{1.f, 2.f}, false);
    const tensor::Tensor<float> b({3}, std::vector<float>{1.f, 2.f, 3.f}, false);
    CHECK_THROWS_AS(ops::add(a, b), std::invalid_argument);
    CHECK_THROWS_AS(ops::multiply(a, b), std::invalid_argument);
    CHECK_THROWS_AS(ops::divide(a, b), std::invalid_argument);
}

TEST_CASE("divide throws on zero and log throws on non-positive") {
    const tensor::Tensor<float> a({1}, std::vector<float>{1.f}, false);
    const tensor::Tensor<float> z({1}, std::vector<float>{0.f}, false);
    CHECK_THROWS_AS(ops::divide(a, z), std::runtime_error);
    CHECK_THROWS_AS(ops::log(z), std::runtime_error);
}

TEST_CASE("abs and relu at zero use a zero local gradient") {
    tensor::Tensor<float> x({3}, std::vector<float>{-1.f, 0.f, 2.f}, true);
    x.abs().sum().backward();
    REQUIRE(x.grad() != nullptr);
    CHECK(x.grad()->data()[0] == doctest::Approx(-1.f));
    CHECK(x.grad()->data()[1] == doctest::Approx(0.f));
    CHECK(x.grad()->data()[2] == doctest::Approx(1.f));

    tensor::Tensor<float> y({3}, std::vector<float>{-1.f, 0.f, 2.f}, true);
    y.relu().sum().backward();
    REQUIRE(y.grad() != nullptr);
    CHECK(y.grad()->data()[0] == doctest::Approx(0.f));
    CHECK(y.grad()->data()[1] == doctest::Approx(0.f));
    CHECK(y.grad()->data()[2] == doctest::Approx(1.f));
}

TEST_CASE("log and sin backward") {
    tensor::Tensor<float> x({2}, std::vector<float>{1.f, 2.f}, true);
    x.log().sum().backward();
    REQUIRE(x.grad() != nullptr);
    CHECK(x.grad()->data()[0] == doctest::Approx(1.f));
    CHECK(x.grad()->data()[1] == doctest::Approx(0.5f));

    tensor::Tensor<float> t({2}, std::vector<float>{0.f, 1.f}, true);
    t.sin().sum().backward();
    REQUIRE(t.grad() != nullptr);
    CHECK(t.grad()->data()[0] == doctest::Approx(std::cos(0.f)));
    CHECK(t.grad()->data()[1] == doctest::Approx(std::cos(1.f)));
}

TEST_CASE("silu backward is sigmoid(x) * (1 + x * (1 - sigmoid(x)))") {
    tensor::Tensor<float> x({1}, std::vector<float>{2.f}, true);
    x.silu().sum().backward();
    const float s = 1.f / (1.f + std::exp(-2.f));
    REQUIRE(x.grad() != nullptr);
    CHECK(x.grad()->data()[0] == doctest::Approx(s * (1.f + 2.f * (1.f - s))));
}
