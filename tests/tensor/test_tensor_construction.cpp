#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <vector>

#include "../doctest/doctest.h"

#include "../../src/tensor/tensor.h"

namespace {

template <typename T>
struct DtypeFor;

template <>
struct DtypeFor<float> {
    static constexpr tensor::Dtype value = tensor::Dtype::Float32;
};

template <>
struct DtypeFor<double> {
    static constexpr tensor::Dtype value = tensor::Dtype::Float64;
};

template <>
struct DtypeFor<int32_t> {
    static constexpr tensor::Dtype value = tensor::Dtype::Int32;
};

template <>
struct DtypeFor<int64_t> {
    static constexpr tensor::Dtype value = tensor::Dtype::Int64;
};

} // namespace

// ============================================================
//  zeros / ones / full
// ============================================================

TEST_CASE_TEMPLATE("Tensor default constructor creates 2x2 zero-filled tensor",
                   TestType,
                   float,
                   double,
                   int32_t,
                   int64_t) {
    const std::vector<int64_t> shape{2, 2};
    const tensor::Tensor<TestType> tensor(shape, false);

    CHECK(tensor.shape() == shape);
    CHECK(tensor.dtype() == DtypeFor<TestType>::value);
    CHECK(tensor.data() == std::vector<TestType>{TestType{0}, TestType{0}, TestType{0}, TestType{0}});
}

TEST_CASE_TEMPLATE("Tensor::zeros creates 2x2 zero-filled tensor",
                   TestType,
                   float,
                   double,
                   int32_t,
                   int64_t) {
    const std::vector<int64_t> shape{2, 2};
    const tensor::Tensor<TestType> tensor = tensor::Tensor<TestType>::zeros(shape, false);

    CHECK(tensor.shape() == shape);
    CHECK(tensor.dtype() == DtypeFor<TestType>::value);
    CHECK(tensor.data() == std::vector<TestType>{TestType{0}, TestType{0}, TestType{0}, TestType{0}});
}

TEST_CASE_TEMPLATE("Tensor::ones creates 2x2 one-filled tensor",
                   TestType,
                   float,
                   double,
                   int32_t,
                   int64_t) {
    const std::vector<int64_t> shape{2, 2};
    const tensor::Tensor<TestType> tensor = tensor::Tensor<TestType>::ones(shape, false);

    CHECK(tensor.shape() == shape);
    CHECK(tensor.dtype() == DtypeFor<TestType>::value);
    CHECK(tensor.data() == std::vector<TestType>{TestType{1}, TestType{1}, TestType{1}, TestType{1}});
}

TEST_CASE_TEMPLATE("Tensor::full creates 2x2 tensor filled with a specific value",
                   TestType,
                   float,
                   double,
                   int32_t,
                   int64_t) {
    const std::vector<int64_t> shape{2, 2};
    constexpr TestType fill_value = static_cast<TestType>(7);
    const tensor::Tensor<TestType> tensor = tensor::Tensor<TestType>::full(shape, fill_value, false);

    CHECK(tensor.shape() == shape);
    CHECK(tensor.dtype() == DtypeFor<TestType>::value);
    CHECK(tensor.data() == std::vector<TestType>{fill_value, fill_value, fill_value, fill_value});
}

// ============================================================
//  zeros_like / ones_like / full_like
// ============================================================

TEST_CASE_TEMPLATE("Tensor::zeros_like produces a zero tensor with matching shape",
                   TestType,
                   float,
                   double,
                   int32_t,
                   int64_t) {
    const std::vector<int64_t> shape{3, 4};
    const tensor::Tensor<TestType> src = tensor::Tensor<TestType>::ones(shape, false);
    const tensor::Tensor<TestType> t   = tensor::Tensor<TestType>::zeros_like(src, false);

    CHECK(t.shape() == shape);
    CHECK(t.dtype() == DtypeFor<TestType>::value);
    for (const auto& v : t.data())
        CHECK(v == static_cast<TestType>(0));
}

TEST_CASE_TEMPLATE("Tensor::ones_like produces a ones tensor with matching shape",
                   TestType,
                   float,
                   double,
                   int32_t,
                   int64_t) {
    const std::vector<int64_t> shape{3, 4};
    const tensor::Tensor<TestType> src = tensor::Tensor<TestType>::zeros(shape, false);
    const tensor::Tensor<TestType> t   = tensor::Tensor<TestType>::ones_like(src, false);

    CHECK(t.shape() == shape);
    CHECK(t.dtype() == DtypeFor<TestType>::value);
    for (const auto& v : t.data())
        CHECK(v == static_cast<TestType>(1));
}

TEST_CASE_TEMPLATE("Tensor::full_like produces a fill tensor with matching shape",
                   TestType,
                   float,
                   double,
                   int32_t,
                   int64_t) {
    const std::vector<int64_t> shape{2, 5};
    constexpr TestType fill_value = static_cast<TestType>(42);
    const tensor::Tensor<TestType> src = tensor::Tensor<TestType>::zeros(shape, false);
    const tensor::Tensor<TestType> t   = tensor::Tensor<TestType>::full_like(src, fill_value, false);

    CHECK(t.shape() == shape);
    CHECK(t.dtype() == DtypeFor<TestType>::value);
    for (const auto& v : t.data())
        CHECK(v == fill_value);
}

// ============================================================
//  randint
// ============================================================

TEST_CASE_TEMPLATE("Tensor::randint produces correct shape and in-range values",
                   TestType,
                   float,
                   double,
                   int32_t,
                   int64_t) {
    const std::vector<int64_t> shape{4, 4};
    constexpr int64_t lo = 0, hi = 10;
    const tensor::Tensor<TestType> t =
        tensor::Tensor<TestType>::randint(shape, lo, hi, false);

    CHECK(t.shape() == shape);
    CHECK(t.dtype() == DtypeFor<TestType>::value);
    CHECK(static_cast<int64_t>(t.data().size()) == tensor::Tensor<TestType>::compute_numel(shape));
    for (const auto& v : t.data()) {
        CHECK(v >= static_cast<TestType>(lo));
        CHECK(v <  static_cast<TestType>(hi));
    }
}

TEST_CASE_TEMPLATE("Tensor::randint with same seed produces identical tensors",
                   TestType,
                   float,
                   double,
                   int32_t,
                   int64_t) {
    const std::vector<int64_t> shape{3, 5};
    constexpr uint64_t seed = 42;
    const auto a = tensor::Tensor<TestType>::randint(shape, -5, 5, false, seed);
    const auto b = tensor::Tensor<TestType>::randint(shape, -5, 5, false, seed);

    CHECK(a.data() == b.data());
}

TEST_CASE_TEMPLATE("Tensor::randint with different seeds produces different tensors",
                   TestType,
                   float,
                   double,
                   int32_t,
                   int64_t) {
    const std::vector<int64_t> shape{1, 64};
    const auto a = tensor::Tensor<TestType>::randint(shape, 0, 100, false, uint64_t{0});
    const auto b = tensor::Tensor<TestType>::randint(shape, 0, 100, false, uint64_t{1});

    CHECK(a.data() != b.data());
}

TEST_CASE_TEMPLATE("Tensor::randint rejects low >= high",
                   TestType,
                   float,
                   double,
                   int32_t,
                   int64_t) {
    CHECK_THROWS_AS(
        tensor::Tensor<TestType>::randint({2, 2}, 5, 5, false),
        std::invalid_argument
    );
    CHECK_THROWS_AS(
        tensor::Tensor<TestType>::randint({2, 2}, 6, 5, false),
        std::invalid_argument
    );
}

TEST_CASE_TEMPLATE("Tensor::randint_like matches source shape",
                   TestType,
                   float,
                   double,
                   int32_t,
                   int64_t) {
    const std::vector<int64_t> shape{2, 3, 4};
    const auto src = tensor::Tensor<TestType>::zeros(shape, false);
    const auto t   = tensor::Tensor<TestType>::randint_like(src, 0, 10, false, uint64_t{7});

    CHECK(t.shape() == shape);
    CHECK(t.dtype() == DtypeFor<TestType>::value);
    for (const auto& v : t.data()) {
        CHECK(v >= static_cast<TestType>(0));
        CHECK(v <  static_cast<TestType>(10));
    }
}

// ============================================================
//  randn
// ============================================================

TEST_CASE_TEMPLATE("Tensor::randn produces correct shape",
                   TestType,
                   float,
                   double) {
    const std::vector<int64_t> shape{5, 6};
    const auto t = tensor::Tensor<TestType>::randn(shape, false);

    CHECK(t.shape() == shape);
    CHECK(t.dtype() == DtypeFor<TestType>::value);
    CHECK(static_cast<int64_t>(t.data().size()) == tensor::Tensor<TestType>::compute_numel(shape));
}

TEST_CASE_TEMPLATE("Tensor::randn with same seed produces identical tensors",
                   TestType,
                   float,
                   double) {
    const std::vector<int64_t> shape{4, 4};
    constexpr uint64_t seed = 99;
    const auto a = tensor::Tensor<TestType>::randn(shape, false, seed);
    const auto b = tensor::Tensor<TestType>::randn(shape, false, seed);

    CHECK(a.data() == b.data());
}

TEST_CASE_TEMPLATE("Tensor::randn with different seeds produces different tensors",
                   TestType,
                   float,
                   double) {
    const std::vector<int64_t> shape{1, 64};
    const auto a = tensor::Tensor<TestType>::randn(shape, false, uint64_t{0});
    const auto b = tensor::Tensor<TestType>::randn(shape, false, uint64_t{1});

    CHECK(a.data() != b.data());
}

TEST_CASE_TEMPLATE("Tensor::randn large sample has near-zero mean and near-unit stddev",
                   TestType,
                   float,
                   double) {
    // 10 000 samples: mean should be within 0.1 and stddev within [0.9, 1.1]
    const std::vector<int64_t> shape{100, 100};
    const auto t = tensor::Tensor<TestType>::randn(shape, false, uint64_t{2024});
    const auto& d = t.data();
    const double n = static_cast<double>(d.size());

    const double mean = std::accumulate(d.begin(), d.end(), 0.0) / n;
    const double var  = std::accumulate(d.begin(), d.end(), 0.0, [&](double acc, TestType v) {
        const double diff = static_cast<double>(v) - mean;
        return acc + diff * diff;
    }) / n;

    CHECK(std::abs(mean) < 0.1);
    CHECK(std::abs(std::sqrt(var) - 1.0) < 0.1);
}

TEST_CASE_TEMPLATE("Tensor::randn_like matches source shape",
                   TestType,
                   float,
                   double) {
    const std::vector<int64_t> shape{3, 7};
    const auto src = tensor::Tensor<TestType>::zeros(shape, false);
    const auto t   = tensor::Tensor<TestType>::randn_like(src, false, uint64_t{5});

    CHECK(t.shape() == shape);
    CHECK(t.dtype() == DtypeFor<TestType>::value);
}

// ============================================================
//  random_gaussian
// ============================================================

TEST_CASE_TEMPLATE("Tensor::random_gaussian produces correct shape",
                   TestType,
                   float,
                   double) {
    const std::vector<int64_t> shape{3, 4};
    const auto t = tensor::Tensor<TestType>::random_gaussian(
        shape, static_cast<TestType>(2), static_cast<TestType>(0.5), false);

    CHECK(t.shape() == shape);
    CHECK(t.dtype() == DtypeFor<TestType>::value);
    CHECK(static_cast<int64_t>(t.data().size()) == tensor::Tensor<TestType>::compute_numel(shape));
}

TEST_CASE_TEMPLATE("Tensor::random_gaussian with same seed produces identical tensors",
                   TestType,
                   float,
                   double) {
    const std::vector<int64_t> shape{4, 4};
    constexpr uint64_t seed = 77;
    const auto a = tensor::Tensor<TestType>::random_gaussian(
        shape, static_cast<TestType>(5), static_cast<TestType>(2), false, seed);
    const auto b = tensor::Tensor<TestType>::random_gaussian(
        shape, static_cast<TestType>(5), static_cast<TestType>(2), false, seed);

    CHECK(a.data() == b.data());
}

TEST_CASE_TEMPLATE("Tensor::random_gaussian large sample has near-correct mean and stddev",
                   TestType,
                   float,
                   double) {
    constexpr TestType target_mean   = static_cast<TestType>(3);
    constexpr TestType target_stddev = static_cast<TestType>(2);
    const std::vector<int64_t> shape{100, 100};
    const auto t = tensor::Tensor<TestType>::random_gaussian(
        shape, target_mean, target_stddev, false, uint64_t{42});
    const auto& d = t.data();
    const double n = static_cast<double>(d.size());

    const double mean = std::accumulate(d.begin(), d.end(), 0.0) / n;
    const double var  = std::accumulate(d.begin(), d.end(), 0.0, [&](double acc, TestType v) {
        const double diff = static_cast<double>(v) - mean;
        return acc + diff * diff;
    }) / n;

    CHECK(std::abs(mean  - static_cast<double>(target_mean))   < 0.1);
    CHECK(std::abs(std::sqrt(var) - static_cast<double>(target_stddev)) < 0.1);
}

TEST_CASE_TEMPLATE("Tensor::random_gaussian_like matches source shape",
                   TestType,
                   float,
                   double) {
    const std::vector<int64_t> shape{4, 5};
    const auto src = tensor::Tensor<TestType>::zeros(shape, false);
    const auto t   = tensor::Tensor<TestType>::random_gaussian_like(
        src, static_cast<TestType>(0), static_cast<TestType>(1), false, uint64_t{8});

    CHECK(t.shape() == shape);
    CHECK(t.dtype() == DtypeFor<TestType>::value);
}

TEST_CASE("shape-only constructor uses global default dtype for matching Tensor<T>") {
    tensor::reset_default_dtype();
    tensor::set_default_dtype(tensor::Dtype::Float32);

    const tensor::Tensor<float> t({2, 2}, false);
    CHECK(t.dtype() == tensor::Dtype::Float32);
}

TEST_CASE("vector constructor infers dtype when dtype is omitted") {
    tensor::reset_default_dtype();
    tensor::set_default_dtype(tensor::Dtype::Float64);

    const std::vector<float> values{1.f, 2.f, 3.f, 4.f};
    const tensor::Tensor<float> t({2, 2}, values, false);
    CHECK(t.dtype() == tensor::Dtype::Float32);
}

TEST_CASE("vector constructor dtype is always inferred from T") {
    const std::vector<float> fvalues{1.f, 2.f, 3.f, 4.f};
    CHECK(tensor::Tensor<float>({2, 2}, fvalues).dtype()   == tensor::Dtype::Float32);

    const std::vector<double> dvalues{1., 2., 3., 4.};
    CHECK(tensor::Tensor<double>({2, 2}, dvalues).dtype()  == tensor::Dtype::Float64);

    const std::vector<int32_t> i32values{1, 2, 3, 4};
    CHECK(tensor::Tensor<int32_t>({2, 2}, i32values).dtype() == tensor::Dtype::Int32);

    const std::vector<int64_t> i64values{1, 2, 3, 4};
    CHECK(tensor::Tensor<int64_t>({2, 2}, i64values).dtype() == tensor::Dtype::Int64);
}

TEST_CASE("shape-only constructor dtype is always inferred from T") {
    CHECK(tensor::Tensor<float>({2, 2}).dtype()   == tensor::Dtype::Float32);
    CHECK(tensor::Tensor<double>({2, 2}).dtype()  == tensor::Dtype::Float64);
    CHECK(tensor::Tensor<int32_t>({2, 2}).dtype() == tensor::Dtype::Int32);
    CHECK(tensor::Tensor<int64_t>({2, 2}).dtype() == tensor::Dtype::Int64);
}

// ============================================================
//  integer dtype disables gradient tracking
// ============================================================

TEST_CASE("Tensor<int32_t> always has requires_grad == false") {
    const tensor::Tensor<int32_t> t({2, 3}, true);
    CHECK_FALSE(t.requires_grad());
}

TEST_CASE("Tensor<int64_t> always has requires_grad == false") {
    const tensor::Tensor<int64_t> t({2, 3}, true);
    CHECK_FALSE(t.requires_grad());
}

TEST_CASE("Tensor<float> preserves requires_grad") {
    const tensor::Tensor<float> t({2, 3}, true);
    CHECK(t.requires_grad());
}

TEST_CASE("vector ctor with Tensor<int32_t> disables requires_grad") {
    const std::vector<int32_t> values{1, 2, 3, 4};
    const tensor::Tensor<int32_t> t({2, 2}, values, true);
    CHECK_FALSE(t.requires_grad());
}

TEST_CASE("zeros factory with Tensor<int64_t> disables requires_grad") {
    const auto t = tensor::Tensor<int64_t>::zeros({3, 3}, true);
    CHECK_FALSE(t.requires_grad());
}

