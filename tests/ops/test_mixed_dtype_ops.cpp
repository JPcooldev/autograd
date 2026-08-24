#include <cmath>
#include <cstdint>
#include <type_traits>
#include <vector>

#include "../doctest/doctest.h"

#include "../../src/autograd/autograd.h"

// ─── to<U>() ──────────────────────────────────────────────────────────────────

TEST_CASE("Tensor::to<double> from float preserves values") {
    const tensor::Tensor<float> a({3}, std::vector<float>{1.5f, -2.0f, 3.25f}, false);
    const auto b = a.to<double>();

    REQUIRE(b.dtype() == tensor::Dtype::Float64);
    REQUIRE(b.shape() == std::vector<int64_t>{3});
    for (size_t i = 0; i < 3; ++i)
        CHECK(b.data()[i] == doctest::Approx(static_cast<double>(a.data()[i])));
}

TEST_CASE("Tensor::to<float> from double truncates precision") {
    const tensor::Tensor<double> a({2}, std::vector<double>{1.0, 2.0}, false);
    const auto b = a.to<float>();

    REQUIRE(b.dtype() == tensor::Dtype::Float32);
    CHECK(b.data()[0] == doctest::Approx(1.0f));
    CHECK(b.data()[1] == doctest::Approx(2.0f));
}

TEST_CASE("Tensor::to<double> from int32_t") {
    const tensor::Tensor<int32_t> a({3}, std::vector<int32_t>{-1, 0, 42}, false);
    const auto b = a.to<double>();

    REQUIRE(b.dtype() == tensor::Dtype::Float64);
    CHECK(b.data()[0] == doctest::Approx(-1.0));
    CHECK(b.data()[1] == doctest::Approx(0.0));
    CHECK(b.data()[2] == doctest::Approx(42.0));
}

TEST_CASE("Tensor::to<double> from float keeps requires_grad") {
    const tensor::Tensor<float> a({2}, std::vector<float>{1.f, 2.f}, true);
    const auto b = a.to<double>();
    CHECK(b.requires_grad());
    CHECK(b.grad_fn().get() != nullptr);
    CHECK_FALSE(b.is_leaf());
}

TEST_CASE("Tensor::to<int32_t> from float is a no-grad leaf") {
    const tensor::Tensor<float> a({2}, std::vector<float>{1.f, 2.f}, true);
    const auto b = a.to<int32_t>();
    CHECK_FALSE(b.requires_grad());
    CHECK(b.grad_fn().get() == nullptr);
}

TEST_CASE("Tensor::to<double> from int32_t is a no-grad leaf") {
    const tensor::Tensor<int32_t> a({2}, std::vector<int32_t>{1, 2}, false);
    const auto b = a.to<double>();
    CHECK_FALSE(b.requires_grad());
    CHECK(b.grad_fn().get() == nullptr);
}

// ─── result type promotion ────────────────────────────────────────────────────

TEST_CASE("mixed-dtype add: float + double -> double") {
    const tensor::Tensor<float>  a({3}, std::vector<float> {1.5f, 2.0f, -1.0f}, false);
    const tensor::Tensor<double> b({3}, std::vector<double>{0.5,  1.0,   3.0 }, false);
    const auto result = ops::add(a, b);

    static_assert(std::is_same<decltype(result), const tensor::Tensor<double>>::value);
    REQUIRE(result.dtype() == tensor::Dtype::Float64);
    REQUIRE(result.shape() == std::vector<int64_t>{3});
    CHECK(result.data()[0] == doctest::Approx(2.0));
    CHECK(result.data()[1] == doctest::Approx(3.0));
    CHECK(result.data()[2] == doctest::Approx(2.0));
}

TEST_CASE("mixed-dtype add: double + float -> double (reversed argument order)") {
    const tensor::Tensor<double> a({2}, std::vector<double>{10.0, 20.0}, false);
    const tensor::Tensor<float>  b({2}, std::vector<float> {0.5f,  0.5f}, false);
    const auto result = ops::add(a, b);

    static_assert(std::is_same<decltype(result), const tensor::Tensor<double>>::value);
    CHECK(result.data()[0] == doctest::Approx(10.5));
    CHECK(result.data()[1] == doctest::Approx(20.5));
}

TEST_CASE("mixed-dtype add: int32_t + float -> float") {
    const tensor::Tensor<int32_t> a({3}, std::vector<int32_t>{1, 2, 3}, false);
    const tensor::Tensor<float>   b({3}, std::vector<float>  {0.5f, 0.5f, 0.5f}, false);
    const auto result = ops::add(a, b);

    static_assert(std::is_same<decltype(result), const tensor::Tensor<float>>::value);
    REQUIRE(result.dtype() == tensor::Dtype::Float32);
    CHECK(result.data()[0] == doctest::Approx(1.5f));
    CHECK(result.data()[1] == doctest::Approx(2.5f));
    CHECK(result.data()[2] == doctest::Approx(3.5f));
}

TEST_CASE("mixed-dtype add: int32_t + int64_t -> int64_t") {
    const tensor::Tensor<int32_t> a({2}, std::vector<int32_t>{10, 20}, false);
    const tensor::Tensor<int64_t> b({2}, std::vector<int64_t>{3,  7 }, false);
    const auto result = ops::add(a, b);

    static_assert(std::is_same<decltype(result), const tensor::Tensor<int64_t>>::value);
    REQUIRE(result.dtype() == tensor::Dtype::Int64);
    CHECK(result.data()[0] == 13);
    CHECK(result.data()[1] == 27);
}

// ─── subtract ─────────────────────────────────────────────────────────────────

TEST_CASE("mixed-dtype subtract: float - double -> double") {
    const tensor::Tensor<float>  a({2}, std::vector<float> {5.0f, 3.0f}, false);
    const tensor::Tensor<double> b({2}, std::vector<double>{1.5,  2.5 }, false);
    const auto result = ops::subtract(a, b);

    static_assert(std::is_same<decltype(result), const tensor::Tensor<double>>::value);
    CHECK(result.data()[0] == doctest::Approx(3.5));
    CHECK(result.data()[1] == doctest::Approx(0.5));
}

// ─── multiply ─────────────────────────────────────────────────────────────────

TEST_CASE("mixed-dtype multiply: int32_t * double -> double") {
    const tensor::Tensor<int32_t> a({2}, std::vector<int32_t>{2, 3}, false);
    const tensor::Tensor<double>  b({2}, std::vector<double> {1.5, 2.0}, false);
    const auto result = ops::multiply(a, b);

    static_assert(std::is_same<decltype(result), const tensor::Tensor<double>>::value);
    CHECK(result.data()[0] == doctest::Approx(3.0));
    CHECK(result.data()[1] == doctest::Approx(6.0));
}

TEST_CASE("mixed-dtype multiply: float * double -> double") {
    const tensor::Tensor<float>  a({2}, std::vector<float> {2.0f, 4.0f}, false);
    const tensor::Tensor<double> b({2}, std::vector<double>{0.5,  0.25 }, false);
    const auto result = ops::multiply(a, b);

    static_assert(std::is_same<decltype(result), const tensor::Tensor<double>>::value);
    CHECK(result.data()[0] == doctest::Approx(1.0));
    CHECK(result.data()[1] == doctest::Approx(1.0));
}

// ─── divide ───────────────────────────────────────────────────────────────────

TEST_CASE("mixed-dtype divide: float / double -> double") {
    const tensor::Tensor<float>  a({2}, std::vector<float> {6.0f, 9.0f}, false);
    const tensor::Tensor<double> b({2}, std::vector<double>{2.0,  3.0 }, false);
    const auto result = ops::divide(a, b);

    static_assert(std::is_same<decltype(result), const tensor::Tensor<double>>::value);
    CHECK(result.data()[0] == doctest::Approx(3.0));
    CHECK(result.data()[1] == doctest::Approx(3.0));
}

TEST_CASE("mixed-dtype divide: int32_t / float -> float") {
    const tensor::Tensor<int32_t> a({2}, std::vector<int32_t>{10, 9}, false);
    const tensor::Tensor<float>   b({2}, std::vector<float>  {4.0f, 3.0f}, false);
    const auto result = ops::divide(a, b);

    static_assert(std::is_same<decltype(result), const tensor::Tensor<float>>::value);
    CHECK(result.data()[0] == doctest::Approx(2.5f));
    CHECK(result.data()[1] == doctest::Approx(3.0f));
}

// ─── shape is preserved ───────────────────────────────────────────────────────

TEST_CASE("mixed-dtype ops preserve shape") {
    const tensor::Tensor<float>  a({2, 3}, std::vector<float> {1,2,3,4,5,6}, false);
    const tensor::Tensor<double> b({2, 3}, std::vector<double>{1,2,3,4,5,6}, false);
    const auto result = ops::add(a, b);

    REQUIRE(result.shape() == (std::vector<int64_t>{2, 3}));
}

// ─── shape mismatch is rejected ───────────────────────────────────────────────

TEST_CASE("mixed-dtype add throws on shape mismatch") {
    const tensor::Tensor<float>  a({2}, std::vector<float> {1.0f, 2.0f}, false);
    const tensor::Tensor<double> b({3}, std::vector<double>{1.0, 2.0, 3.0}, false);
    CHECK_THROWS_AS(ops::add(a, b), std::invalid_argument);
}

// ─── named cast wrappers ──────────────────────────────────────────────────────

TEST_CASE("float32() casts values and keeps requires_grad") {
    const tensor::Tensor<double> a({3}, std::vector<double>{1.5, -2.0, 4.0}, true);
    const auto b = a.float32();

    REQUIRE(b.dtype() == tensor::Dtype::Float32);
    CHECK(b.requires_grad());
    CHECK(b.grad_fn().get() != nullptr);
    CHECK(b.data()[0] == doctest::Approx(1.5f));
    CHECK(b.data()[1] == doctest::Approx(-2.0f));
    CHECK(b.data()[2] == doctest::Approx(4.0f));
}

TEST_CASE("float64() casts values and returns a no-grad Float64 leaf") {
    const tensor::Tensor<int32_t> a({2}, std::vector<int32_t>{3, -7}, false);
    const auto b = a.float64();

    REQUIRE(b.dtype() == tensor::Dtype::Float64);
    CHECK_FALSE(b.requires_grad());
    CHECK(b.data()[0] == doctest::Approx(3.0));
    CHECK(b.data()[1] == doctest::Approx(-7.0));
}

TEST_CASE("int32() truncates toward zero") {
    const tensor::Tensor<float> a({3}, std::vector<float>{1.9f, -2.7f, 0.1f}, true);
    const auto b = a.int32();

    REQUIRE(b.dtype() == tensor::Dtype::Int32);
    CHECK_FALSE(b.requires_grad());
    CHECK(b.data() == std::vector<int32_t>{1, -2, 0});
}

TEST_CASE("int64() from float64") {
    const tensor::Tensor<double> a({2}, std::vector<double>{42.9, -1.1}, false);
    const auto b = a.int64();

    REQUIRE(b.dtype() == tensor::Dtype::Int64);
    CHECK(b.data() == std::vector<int64_t>{42, -1});
}

TEST_CASE("to<U>() copies a transposed view in logical order") {
    const tensor::Tensor<float> a({2, 3}, std::vector<float>{1, 2, 3, 4, 5, 6}, false);
    const auto t = a.transpose();
    const auto b = t.to<double>();

    REQUIRE(b.shape() == (std::vector<int64_t>{3, 2}));
    CHECK(b.data() == (std::vector<double>{1, 4, 2, 5, 3, 6}));
}
