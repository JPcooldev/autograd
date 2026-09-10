/*
 * Embedding lookup and scatter-add backward.
 *
 * - gather rows of a {V, D} table
 * - rank-2 indices → rank-3 output
 * - int64 indices
 * - OOB (too large or negative) and non-rank-2 weight throw
 * - repeated indices scatter-add into weight.grad
 */

#include <cstdint>
#include <stdexcept>
#include <vector>

#include "../doctest/doctest.h"

#include "../../src/autograd/autograd.h"
#include "../../src/ops/embedding_ops.h"

TEST_CASE("embedding gathers rows of a {V, D} table") {
    tensor::Tensor<float> weight(
        {4, 3},
        std::vector<float>{
            1.f, 2.f, 3.f,
            4.f, 5.f, 6.f,
            7.f, 8.f, 9.f,
            10.f, 11.f, 12.f
        },
        false);
    tensor::Tensor<int32_t> idx({2}, std::vector<int32_t>{3, 0}, false);
    auto y = ops::embedding(weight, idx);
    CHECK(y.shape() == std::vector<int64_t>{2, 3});
    CHECK(y.data() == std::vector<float>{10.f, 11.f, 12.f, 1.f, 2.f, 3.f});
}

TEST_CASE("embedding rank-2 indices yield rank-3 output") {
    tensor::Tensor<float> weight({3, 2}, std::vector<float>{1.f, 2.f, 3.f, 4.f, 5.f, 6.f}, false);
    tensor::Tensor<int32_t> idx({2, 2}, std::vector<int32_t>{0, 2, 1, 0}, false);
    auto y = ops::embedding(weight, idx);
    CHECK(y.shape() == std::vector<int64_t>{2, 2, 2});
    CHECK(y.data()[0] == doctest::Approx(1.f));
    CHECK(y.data()[1] == doctest::Approx(2.f));
    CHECK(y.data()[2] == doctest::Approx(5.f));
    CHECK(y.data()[3] == doctest::Approx(6.f));
    CHECK(y.data()[4] == doctest::Approx(3.f));
    CHECK(y.data()[5] == doctest::Approx(4.f));
    CHECK(y.data()[6] == doctest::Approx(1.f));
    CHECK(y.data()[7] == doctest::Approx(2.f));
}

TEST_CASE("embedding accepts int64 indices") {
    tensor::Tensor<float> weight({2, 1}, std::vector<float>{9.f, 8.f}, false);
    tensor::Tensor<int64_t> idx({1}, std::vector<int64_t>{1}, false);
    auto y = ops::embedding(weight, idx);
    CHECK(y.data()[0] == doctest::Approx(8.f));
}

TEST_CASE("embedding throws on out-of-range index") {
    tensor::Tensor<float> weight({2, 1}, std::vector<float>{1.f, 2.f}, false);
    tensor::Tensor<int32_t> idx({1}, std::vector<int32_t>{2}, false);
    CHECK_THROWS_AS(ops::embedding(weight, idx), std::invalid_argument);
    tensor::Tensor<int32_t> neg({1}, std::vector<int32_t>{-1}, false);
    CHECK_THROWS_AS(ops::embedding(weight, neg), std::invalid_argument);
}

TEST_CASE("embedding throws if weight is not rank 2") {
    tensor::Tensor<float> weight({4}, std::vector<float>{1.f, 2.f, 3.f, 4.f}, false);
    tensor::Tensor<int32_t> idx({1}, std::vector<int32_t>{0}, false);
    CHECK_THROWS_AS(ops::embedding(weight, idx), std::invalid_argument);
}

TEST_CASE("EmbeddingBackward scatter-adds repeated indices") {
    tensor::Tensor<float> weight({3, 2}, std::vector<float>{0.f, 0.f, 0.f, 0.f, 0.f, 0.f}, true);
    tensor::Tensor<int32_t> idx({3}, std::vector<int32_t>{1, 0, 1}, false);
    auto y = ops::embedding(weight, idx);
    y.sum().backward();
    REQUIRE(weight.grad() != nullptr);
    // each output element contributes 1; rows: 0 once, 1 twice, 2 never
    CHECK(weight.grad()->data()[0] == doctest::Approx(1.f));
    CHECK(weight.grad()->data()[1] == doctest::Approx(1.f));
    CHECK(weight.grad()->data()[2] == doctest::Approx(2.f));
    CHECK(weight.grad()->data()[3] == doctest::Approx(2.f));
    CHECK(weight.grad()->data()[4] == doctest::Approx(0.f));
    CHECK(weight.grad()->data()[5] == doctest::Approx(0.f));
}
