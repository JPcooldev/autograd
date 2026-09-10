/*
 * Tensor constructors, factories, leaf grad storage, and layout helpers.
 *
 * - zero / vector / raw-array constructors
 * - zeros, ones, full and *_like
 * - arange (step, empty, step==0 throw)
 * - randn / random_gaussian / *_like statistics
 * - xavier / kaiming uniform and normal bounds
 * - uniform in [low, high)
 * - floating leaves allocate grad storage; integer leaves never do
 * - requires_grad forced off for integer dtypes
 * - arithmetic result owns storage (grad_fn on floats); transpose/reshape is a view without grad storage
 * - 0-dim (scalar) and empty tensors
 * - default requires_grad on float
 * - thread-local default dtype get/set/reset (ignored by Tensor<T>)
 * - negative dim throws
 * - randint range and low>=high throw
 * - compute_fans for linear, conv, and rank-1
 */

#include <cmath>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "../doctest/doctest.h"

#include "../../src/autograd/autograd.h"

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

/**
 * Compute the arithmetic mean of `data` in double precision.
 *
 * @param data Sample values.
 * @return `sum(data) / data.size()`.
 */
template <typename T>
double sample_mean(const std::vector<T>& data) {
    return std::accumulate(data.begin(), data.end(), 0.0)
         / static_cast<double>(data.size());
}

/**
 * Compute the population standard deviation of `data`.
 * Uses `sqrt(sum((x - mean)^2) / n)` with `mean` from `sample_mean`.
 *
 * @param data Sample values.
 * @return Population stddev, or NaN if `data` is empty.
 */
template <typename T>
double sample_stddev(const std::vector<T>& data) {
    const double mean = sample_mean(data);
    double acc = 0.0;
    for (const auto v : data) {
        const double diff = static_cast<double>(v) - mean;
        acc += diff * diff;
    }
    return std::sqrt(acc / static_cast<double>(data.size()));
}

// GradStorage is private. Presence is observed through accumulate_grad():
// floating leaves store a matching tensor; integer leaves stay at nullptr.
/**
 * Assert leaf gradient bookkeeping after accumulating ones.
 * Floating-point leaves keep `requires_grad` and allocate a matching grad;
 * integer leaves stay no-grad with a null grad pointer.
 *
 * @param t Leaf tensor to check (taken by value).
 */
template <typename T>
void check_leaf_grad_storage(tensor::Tensor<T> t) {
    CHECK(t.grad_fn().get() == nullptr);
    CHECK(t.grad() == nullptr);

    const std::vector<T> ones(static_cast<size_t>(t.numel()), T{1});
    t.accumulate_grad(tensor::Tensor<T>(t.shape(), ones, false));

    if (std::is_floating_point<T>::value) {
        CHECK(t.requires_grad());
        REQUIRE(t.grad() != nullptr);
        CHECK(t.grad()->shape() == t.shape());
        REQUIRE(t.grad()->data().size() == ones.size());
        for (size_t i = 0; i < ones.size(); ++i)
            CHECK(t.grad()->data()[i] == ones[i]);
    } else {
        CHECK_FALSE(t.requires_grad());
        CHECK(t.grad() == nullptr);
    }
}

/**
 * Assert that `accumulate_grad` does not allocate grad storage.
 *
 * @param t Tensor with `requires_grad == false` (taken by value).
 */
template <typename T>
void check_no_grad_storage(tensor::Tensor<T> t) {
    CHECK(t.grad() == nullptr);
    t.accumulate_grad(tensor::Tensor<T>::ones(t.shape(), false));
    CHECK(t.grad() == nullptr);
}

} // namespace

// ============================================================
//  zero-initialized construction
// ============================================================

TEST_CASE_TEMPLATE(
    "zero-initialized constructor sets shape, dtype, and zeros",
    TestType,
    float,
    double,
    int32_t,
    int64_t
) {
    const std::vector<int64_t> shape{3, 4};
    const tensor::Tensor<TestType> t(shape, false);

    CHECK(t.shape() == shape);
    CHECK(t.dtype() == DtypeFor<TestType>::value);
    CHECK(t.data() == std::vector<TestType>(12, TestType{0}));
}

// ============================================================
//  construction from a vector
// ============================================================

TEST_CASE_TEMPLATE(
    "vector constructor copies shape, dtype, and values",
    TestType,
    float,
    double,
    int32_t,
    int64_t
) {
    const std::vector<int64_t> shape{2, 3};
    const std::vector<TestType> values{
        TestType{1}, TestType{2}, TestType{3},
        TestType{4}, TestType{5}, TestType{6}
    };
    const tensor::Tensor<TestType> t(shape, values, false);

    CHECK(t.shape() == shape);
    CHECK(t.dtype() == DtypeFor<TestType>::value);
    CHECK(t.data() == values);
}

// ============================================================
//  construction from a raw array
// ============================================================

TEST_CASE_TEMPLATE(
    "raw-array constructor copies shape, dtype, and values",
    TestType,
    float,
    double,
    int32_t,
    int64_t
) {
    const std::vector<int64_t> shape{2, 2};
    const TestType arr[4] = {TestType{10}, TestType{20}, TestType{30}, TestType{40}};
    const tensor::Tensor<TestType> t(shape, arr, 4, false);

    CHECK(t.shape() == shape);
    CHECK(t.dtype() == DtypeFor<TestType>::value);
    CHECK(t.data() == std::vector<TestType>(arr, arr + 4));
}

// ============================================================
//  fill initializations
// ============================================================

TEST_CASE_TEMPLATE(
    "zeros / ones / full set shape, dtype, and fill value",
    TestType,
    float,
    double,
    int32_t,
    int64_t
) {
    const std::vector<int64_t> shape{2, 2};
    constexpr TestType fill = static_cast<TestType>(7);

    const auto z = tensor::Tensor<TestType>::zeros(shape, false);
    CHECK(z.shape() == shape);
    CHECK(z.dtype() == DtypeFor<TestType>::value);
    CHECK(z.data() == std::vector<TestType>(4, TestType{0}));

    const auto o = tensor::Tensor<TestType>::ones(shape, false);
    CHECK(o.shape() == shape);
    CHECK(o.dtype() == DtypeFor<TestType>::value);
    CHECK(o.data() == std::vector<TestType>(4, TestType{1}));

    const auto f = tensor::Tensor<TestType>::full(shape, fill, false);
    CHECK(f.shape() == shape);
    CHECK(f.dtype() == DtypeFor<TestType>::value);
    CHECK(f.data() == std::vector<TestType>(4, fill));
}

TEST_CASE_TEMPLATE(
    "zeros_like / ones_like / full_like match source shape and fill",
    TestType,
    float,
    double,
    int32_t,
    int64_t
) {
    const std::vector<int64_t> shape{3, 4};
    constexpr TestType fill = static_cast<TestType>(42);
    const auto src = tensor::Tensor<TestType>::ones(shape, false);

    const auto z = tensor::Tensor<TestType>::zeros_like(src, false);
    CHECK(z.shape() == shape);
    CHECK(z.dtype() == DtypeFor<TestType>::value);
    CHECK(z.data() == std::vector<TestType>(12, TestType{0}));

    const auto o = tensor::Tensor<TestType>::ones_like(src, false);
    CHECK(o.shape() == shape);
    CHECK(o.dtype() == DtypeFor<TestType>::value);
    CHECK(o.data() == std::vector<TestType>(12, TestType{1}));

    const auto f = tensor::Tensor<TestType>::full_like(src, fill, false);
    CHECK(f.shape() == shape);
    CHECK(f.dtype() == DtypeFor<TestType>::value);
    CHECK(f.data() == std::vector<TestType>(12, fill));
}

TEST_CASE_TEMPLATE(
    "arange fills [start, end) with ceiling length",
    TestType,
    float,
    double,
    int32_t,
    int64_t
) {
    const auto exact = tensor::Tensor<TestType>::arange(0, 5, 1, false);
    CHECK(exact.shape() == std::vector<int64_t>{5});
    CHECK(exact.dtype() == DtypeFor<TestType>::value);
    CHECK(exact.data() == std::vector<TestType>{0, 1, 2, 3, 4});

    const auto stepped = tensor::Tensor<TestType>::arange(0, 5, 2, false);
    CHECK(stepped.shape() == std::vector<int64_t>{3});
    CHECK(stepped.data() == std::vector<TestType>{0, 2, 4});

    const auto negative = tensor::Tensor<TestType>::arange(5, 0, -2, false);
    CHECK(negative.shape() == std::vector<int64_t>{3});
    CHECK(negative.data() == std::vector<TestType>{5, 3, 1});

    const auto empty_forward = tensor::Tensor<TestType>::arange(5, 0, 1, false);
    CHECK(empty_forward.shape() == std::vector<int64_t>{0});
    CHECK(empty_forward.data().empty());

    const auto empty_backward = tensor::Tensor<TestType>::arange(0, 5, -1, false);
    CHECK(empty_backward.shape() == std::vector<int64_t>{0});
    CHECK(empty_backward.data().empty());

    CHECK_THROWS_AS(
        tensor::Tensor<TestType>::arange(0, 5, 0, false),
        std::invalid_argument);
}

// ============================================================
//  random initialization
// ============================================================

TEST_CASE_TEMPLATE(
    "randn has shape, dtype, and N(0, 1) statistics",
    TestType,
    float,
    double
) {
    const std::vector<int64_t> shape{100, 100};
    const auto t = tensor::Tensor<TestType>::randn(shape, false, uint64_t{2024});

    CHECK(t.shape() == shape);
    CHECK(t.dtype() == DtypeFor<TestType>::value);
    CHECK(std::abs(sample_mean(t.data())) < 0.1);
    CHECK(std::abs(sample_stddev(t.data()) - 1.0) < 0.1);
}

TEST_CASE_TEMPLATE(
    "random_gaussian has shape, dtype, and N(mean, std) statistics",
    TestType,
    float,
    double
) {
    constexpr TestType mean = static_cast<TestType>(3);
    constexpr TestType stddev = static_cast<TestType>(2);
    const std::vector<int64_t> shape{100, 100};
    const auto t = tensor::Tensor<TestType>::random_gaussian(
        shape, mean, stddev, false, uint64_t{42});

    CHECK(t.shape() == shape);
    CHECK(t.dtype() == DtypeFor<TestType>::value);
    CHECK(std::abs(sample_mean(t.data()) - static_cast<double>(mean)) < 0.1);
    CHECK(std::abs(sample_stddev(t.data()) - static_cast<double>(stddev)) < 0.1);
}

TEST_CASE_TEMPLATE(
    "randn_like matches source shape and N(0, 1) statistics",
    TestType,
    float,
    double
) {
    const std::vector<int64_t> shape{80, 80};
    const auto src = tensor::Tensor<TestType>::zeros(shape, false);
    const auto t = tensor::Tensor<TestType>::randn_like(src, false, uint64_t{5});

    CHECK(t.shape() == shape);
    CHECK(t.dtype() == DtypeFor<TestType>::value);
    CHECK(std::abs(sample_mean(t.data())) < 0.1);
    CHECK(std::abs(sample_stddev(t.data()) - 1.0) < 0.1);
}

TEST_CASE_TEMPLATE(
    "random_gaussian_like matches source shape and N(mean, std) statistics",
    TestType,
    float,
    double
) {
    constexpr TestType mean = static_cast<TestType>(-1);
    constexpr TestType stddev = static_cast<TestType>(0.5);
    const std::vector<int64_t> shape{80, 80};
    const auto src = tensor::Tensor<TestType>::zeros(shape, false);
    const auto t = tensor::Tensor<TestType>::random_gaussian_like(
        src, mean, stddev, false, uint64_t{8});

    CHECK(t.shape() == shape);
    CHECK(t.dtype() == DtypeFor<TestType>::value);
    CHECK(std::abs(sample_mean(t.data()) - static_cast<double>(mean)) < 0.1);
    CHECK(std::abs(sample_stddev(t.data()) - static_cast<double>(stddev)) < 0.1);
}

TEST_CASE_TEMPLATE(
    "xavier_uniform is bounded by Glorot limit with near-zero mean",
    TestType,
    float,
    double
) {
    const std::vector<int64_t> shape{100, 100};
    const auto [fan_in, fan_out] = tensor::Tensor<TestType>::compute_fans(shape);
    const double limit = std::sqrt(6.0 / static_cast<double>(fan_in + fan_out));
    const auto t = tensor::Tensor<TestType>::xavier_uniform(shape, 1.0, false, uint64_t{11});

    CHECK(t.shape() == shape);
    CHECK(t.dtype() == DtypeFor<TestType>::value);
    for (const auto v : t.data())
        CHECK(std::abs(static_cast<double>(v)) <= limit + 1e-6);
    CHECK(std::abs(sample_mean(t.data())) < 0.05);
}

TEST_CASE_TEMPLATE(
    "xavier_normal has shape, dtype, and Glorot N(0, std) statistics",
    TestType,
    float,
    double
) {
    const std::vector<int64_t> shape{100, 100};
    const auto [fan_in, fan_out] = tensor::Tensor<TestType>::compute_fans(shape);
    const double std = std::sqrt(2.0 / static_cast<double>(fan_in + fan_out));
    const auto t = tensor::Tensor<TestType>::xavier_normal(shape, 1.0, false, uint64_t{12});

    CHECK(t.shape() == shape);
    CHECK(t.dtype() == DtypeFor<TestType>::value);
    CHECK(std::abs(sample_mean(t.data())) < 0.05);
    CHECK(std::abs(sample_stddev(t.data()) - std) < 0.05);
}

TEST_CASE_TEMPLATE(
    "kaiming_uniform is bounded by He limit with near-zero mean",
    TestType,
    float,
    double
) {
    const std::vector<int64_t> shape{100, 100};
    const int64_t fan_in = tensor::Tensor<TestType>::compute_fans(shape).first;
    const double gain = std::sqrt(2.0);
    const double bound = std::sqrt(3.0) * gain / std::sqrt(static_cast<double>(fan_in));
    const auto t = tensor::Tensor<TestType>::kaiming_uniform(
        shape, 0.0, "fan_in", false, uint64_t{13});

    CHECK(t.shape() == shape);
    CHECK(t.dtype() == DtypeFor<TestType>::value);
    for (const auto v : t.data())
        CHECK(std::abs(static_cast<double>(v)) <= bound + 1e-6);
    CHECK(std::abs(sample_mean(t.data())) < 0.05);
}

TEST_CASE_TEMPLATE(
    "kaiming_normal has shape, dtype, and He N(0, std) statistics",
    TestType,
    float,
    double
) {
    const std::vector<int64_t> shape{100, 100};
    const int64_t fan_in = tensor::Tensor<TestType>::compute_fans(shape).first;
    const double std = std::sqrt(2.0) / std::sqrt(static_cast<double>(fan_in));
    const auto t = tensor::Tensor<TestType>::kaiming_normal(
        shape, 0.0, "fan_in", false, uint64_t{14});

    CHECK(t.shape() == shape);
    CHECK(t.dtype() == DtypeFor<TestType>::value);
    CHECK(std::abs(sample_mean(t.data())) < 0.05);
    CHECK(std::abs(sample_stddev(t.data()) - std) < 0.05);
}

TEST_CASE_TEMPLATE(
    "uniform has shape, dtype, values in [low, high), and matching mean",
    TestType,
    float,
    double
) {
    constexpr TestType low = static_cast<TestType>(-2);
    constexpr TestType high = static_cast<TestType>(4);
    const std::vector<int64_t> shape{100, 100};
    const auto t = tensor::Tensor<TestType>::uniform(
        shape, low, high, false, uint64_t{15});

    CHECK(t.shape() == shape);
    CHECK(t.dtype() == DtypeFor<TestType>::value);
    for (const auto v : t.data()) {
        CHECK(v >= low);
        CHECK(v < high);
    }
    const double expected_mean =
        0.5 * (static_cast<double>(low) + static_cast<double>(high));
    CHECK(std::abs(sample_mean(t.data()) - expected_mean) < 0.1);
}

// ============================================================
//  leaf grad storage: floats have it, ints do not
// ============================================================

TEST_CASE_TEMPLATE(
    "zero-initialized leaf has grad storage only for floating dtypes",
    TestType,
    float,
    double,
    int32_t,
    int64_t
) {
    check_leaf_grad_storage(tensor::Tensor<TestType>({2, 3}, true));
}

TEST_CASE_TEMPLATE(
    "vector-constructed leaf has grad storage only for floating dtypes",
    TestType,
    float,
    double,
    int32_t,
    int64_t
) {
    const std::vector<TestType> values{1, 2, 3, 4, 5, 6};
    check_leaf_grad_storage(tensor::Tensor<TestType>({2, 3}, values, true));
}

TEST_CASE_TEMPLATE(
    "raw-array leaf has grad storage only for floating dtypes",
    TestType,
    float,
    double,
    int32_t,
    int64_t
) {
    const TestType arr[6] = {1, 2, 3, 4, 5, 6};
    check_leaf_grad_storage(tensor::Tensor<TestType>({2, 3}, arr, 6, true));
}

TEST_CASE_TEMPLATE(
    "fill constructors have grad storage only for floating dtypes",
    TestType,
    float,
    double,
    int32_t,
    int64_t
) {
    const std::vector<int64_t> shape{2, 3};
    const auto src = tensor::Tensor<TestType>::zeros(shape, false);

    check_leaf_grad_storage(tensor::Tensor<TestType>::zeros(shape, true));
    check_leaf_grad_storage(tensor::Tensor<TestType>::ones(shape, true));
    check_leaf_grad_storage(tensor::Tensor<TestType>::full(shape, TestType{3}, true));
    check_leaf_grad_storage(tensor::Tensor<TestType>::zeros_like(src, true));
    check_leaf_grad_storage(tensor::Tensor<TestType>::ones_like(src, true));
    check_leaf_grad_storage(tensor::Tensor<TestType>::full_like(src, TestType{3}, true));
}

// ============================================================
//  requires_grad is forced off for integer T
// ============================================================

TEST_CASE_TEMPLATE(
    "constructors with requires_grad force it off for integer dtypes",
    TestType,
    int32_t,
    int64_t
) {
    const std::vector<int64_t> shape{2, 2};
    const std::vector<TestType> values{1, 2, 3, 4};
    const TestType arr[4] = {1, 2, 3, 4};
    const auto src = tensor::Tensor<TestType>::zeros(shape, false);

    CHECK_FALSE(tensor::Tensor<TestType>(shape, true).requires_grad());
    CHECK_FALSE(tensor::Tensor<TestType>(shape, values, true).requires_grad());
    CHECK_FALSE(tensor::Tensor<TestType>(shape, arr, 4, true).requires_grad());
    CHECK_FALSE(tensor::Tensor<TestType>(shape, TestType{1}, true).requires_grad());
    CHECK_FALSE(tensor::Tensor<TestType>::zeros(shape, true).requires_grad());
    CHECK_FALSE(tensor::Tensor<TestType>::ones(shape, true).requires_grad());
    CHECK_FALSE(tensor::Tensor<TestType>::full(shape, TestType{3}, true).requires_grad());
    CHECK_FALSE(tensor::Tensor<TestType>::zeros_like(src, true).requires_grad());
    CHECK_FALSE(tensor::Tensor<TestType>::ones_like(src, true).requires_grad());
    CHECK_FALSE(tensor::Tensor<TestType>::full_like(src, TestType{3}, true).requires_grad());
    CHECK_FALSE(tensor::Tensor<TestType>::randn(shape, true).requires_grad());
    CHECK_FALSE(tensor::Tensor<TestType>::random_gaussian(
        shape, TestType{0}, TestType{1}, true).requires_grad());
    CHECK_FALSE(tensor::Tensor<TestType>::randn_like(src, true).requires_grad());
    CHECK_FALSE(tensor::Tensor<TestType>::random_gaussian_like(
        src, TestType{0}, TestType{1}, true).requires_grad());
    CHECK_FALSE(tensor::Tensor<TestType>::xavier_uniform(shape, 1.0, true).requires_grad());
    CHECK_FALSE(tensor::Tensor<TestType>::xavier_normal(shape, 1.0, true).requires_grad());
    CHECK_FALSE(tensor::Tensor<TestType>::kaiming_uniform(
        shape, 0.0, "fan_in", true).requires_grad());
    CHECK_FALSE(tensor::Tensor<TestType>::kaiming_normal(
        shape, 0.0, "fan_in", true).requires_grad());
    CHECK_FALSE(tensor::Tensor<TestType>::uniform(
        shape, TestType{0}, TestType{1}, true).requires_grad());
}

// ============================================================
//  arithmetic vs shape operations
// ============================================================

TEST_CASE_TEMPLATE(
    "arithmetic op result owns new storage and has no grad storage",
    TestType,
    float,
    double,
    int32_t,
    int64_t
) {
    tensor::Tensor<TestType> a({2, 2}, std::vector<TestType>{1, 2, 3, 4}, true);
    tensor::Tensor<TestType> b({2, 2}, std::vector<TestType>{5, 6, 7, 8}, true);
    auto c = a.add(b);

    CHECK(c.shape() == a.shape());
    CHECK(c.data() == std::vector<TestType>{6, 8, 10, 12});
    CHECK(c.data().data() != a.data().data());
    CHECK(c.data().data() != b.data().data());
    CHECK_FALSE(c.is_view());
    if (std::is_floating_point<TestType>::value)
        CHECK(c.grad_fn().get() != nullptr);
    else
        CHECK(c.grad_fn().get() == nullptr);
    check_no_grad_storage(c);
}

TEST_CASE_TEMPLATE(
    "shape op result is a view sharing storage with no grad storage",
    TestType,
    float,
    double,
    int32_t,
    int64_t
) {
    tensor::Tensor<TestType> a(
        {2, 3}, std::vector<TestType>{1, 2, 3, 4, 5, 6}, true);

    auto transposed = a.transpose();
    CHECK(transposed.is_view());
    CHECK(transposed.data().data() == a.data().data());
    CHECK(transposed.shape() == std::vector<int64_t>{3, 2});
    check_no_grad_storage(transposed);

    auto reshaped = a.reshape({3, 2});
    CHECK(reshaped.is_view());
    CHECK(reshaped.data().data() == a.data().data());
    CHECK(reshaped.shape() == std::vector<int64_t>{3, 2});
    check_no_grad_storage(reshaped);
    if (std::is_floating_point<TestType>::value) {
        CHECK(transposed.requires_grad());
        CHECK(transposed.grad_fn().get() != nullptr);
        CHECK_FALSE(transposed.is_leaf());
        CHECK(reshaped.requires_grad());
        CHECK(reshaped.grad_fn().get() != nullptr);
        CHECK_FALSE(reshaped.is_leaf());
    }
}

TEST_CASE("0-dim tensor has rank 0, numel 1, and empty strides") {
    const tensor::Tensor<float> s({}, std::vector<float>{3.f}, false);
    CHECK(s.rank() == 0);
    CHECK(s.numel() == 1);
    CHECK(s.strides().empty());
    CHECK(s.is_contiguous());
    CHECK(s.to_vector() == std::vector<float>{3.f});
}

TEST_CASE("empty tensor has numel 0") {
    const tensor::Tensor<float> e({0}, false);
    CHECK(e.numel() == 0);
    CHECK(e.to_vector().empty());
}

TEST_CASE("float constructor defaults to requires_grad") {
    const tensor::Tensor<float> t({2, 2});
    CHECK(t.requires_grad());
    CHECK(t.is_leaf());
}

TEST_CASE("default dtype get/set/reset is thread-local and ignored by Tensor<T>") {
    CHECK(tensor::get_default_dtype() == tensor::Dtype::Float32);
    tensor::set_default_dtype(tensor::Dtype::Float64);
    CHECK(tensor::get_default_dtype() == tensor::Dtype::Float64);
    const tensor::Tensor<float> t({1}, std::vector<float>{1.f}, false);
    CHECK(t.dtype() == tensor::Dtype::Float32);
    tensor::reset_default_dtype();
    CHECK(tensor::get_default_dtype() == tensor::Dtype::Float32);
}

TEST_CASE("negative shape dimension throws") {
    CHECK_THROWS_AS((tensor::Tensor<float>({-1})), std::invalid_argument);
}

TEST_CASE("randint is always no-grad and in [low, high)") {
    auto t = tensor::Tensor<int32_t>::randint({20}, 2, 5);
    CHECK_FALSE(t.requires_grad());
    CHECK(t.shape() == std::vector<int64_t>{20});
    for (int32_t v : t.data()) {
        CHECK(v >= 2);
        CHECK(v < 5);
    }
    CHECK_THROWS_AS(tensor::Tensor<int32_t>::randint({2}, 5, 5), std::invalid_argument);
}

TEST_CASE("compute_fans on linear, conv, and rank-1 shapes") {
    auto linear = tensor::Tensor<float>::compute_fans({8, 32});
    CHECK(linear.first == 32);
    CHECK(linear.second == 8);
    auto conv = tensor::Tensor<float>::compute_fans({16, 3, 5, 5});
    CHECK(conv.first == 75);
    CHECK(conv.second == 400);
    auto vec = tensor::Tensor<float>::compute_fans({3});
    CHECK(vec.first == 3);
    CHECK(vec.second == 3);
}
