/*
 * Reductions: sum, mean, max, min, softmax — forward, dim, and backward.
 *
 * - sum: 1-D / 2-D / single element; empty throw; no-grad; SumBackward node
 * - SumBackward broadcasts a scalar uniformly
 * - sum(dim): rows, cols, negative axis, 3-D middle, rank-1 → scalar
 * - sum(dim) throws on scalar and OOB dim; SumDimBackward
 * - mean global and MeanBackward 1/numel
 * - mean(dim): rows, cols, negative axis; scalar / empty-dim throw; MeanDimBackward
 * - max/min global; empty throw; scatter to argmax/argmin; first-tie
 * - max/min(dim): 2-D, negative, 3-D, rank-1; scalar / OOB / empty-dim throw
 * - MaxDim/MinDim backward and first-tie along dim
 * - no grad_fn when inputs are no-grad; next_edge wiring
 * - softmax last axis, non-last axis, JVP formula
 * - sum of transpose / broadcast / narrow uses logical elements
 */

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "../doctest/doctest.h"

#include "../../src/autograd/autograd.h"

// ─── sum() global ─────────────────────────────────────────────────────────────

TEST_CASE("sum() reduces 1-D tensor to scalar") {
    const tensor::Tensor<float> x({4}, std::vector<float>{1.f, 2.f, 3.f, 4.f}, false);
    const auto result = x.sum();

    CHECK(result.shape() == std::vector<int64_t>{});
    CHECK(result.numel() == 1);
    CHECK(result.data()[0] == doctest::Approx(10.f));
}

TEST_CASE("sum() reduces 2-D tensor to scalar") {
    // [[1, 2, 3], [4, 5, 6]] -> 21
    const tensor::Tensor<float> x({2, 3}, std::vector<float>{1.f, 2.f, 3.f, 4.f, 5.f, 6.f}, false);
    const auto result = x.sum();

    CHECK(result.shape() == std::vector<int64_t>{});
    CHECK(result.data()[0] == doctest::Approx(21.f));
}

TEST_CASE("sum() on a single-element tensor returns that element") {
    const tensor::Tensor<double> x({1}, std::vector<double>{7.5}, false);
    CHECK(x.sum().data()[0] == doctest::Approx(7.5));
}

TEST_CASE("sum() throws on empty tensor") {
    const tensor::Tensor<float> x({0}, false);
    CHECK_THROWS_AS(x.sum(), std::invalid_argument);
}

TEST_CASE("sum() does not track grad when requires_grad=false") {
    const tensor::Tensor<float> x({3}, std::vector<float>{1.f, 2.f, 3.f}, false);
    const auto result = x.sum();
    CHECK_FALSE(result.requires_grad());
    CHECK(result.grad_fn().get() == nullptr);
}

TEST_CASE("sum() tracks grad and creates SumBackward node") {
    const tensor::Tensor<float> x({3}, std::vector<float>{1.f, 2.f, 3.f}, true);
    const auto result = x.sum();

    REQUIRE(result.requires_grad());
    REQUIRE(result.grad_fn().get() != nullptr);
    REQUIRE(result.grad_fn()->next_edges.size() == 1);
    // x is a leaf: with AccumulateGrad eliminated, the edge is nullptr and the
    // engine accumulates directly via saved_tensors[0].
    CHECK(result.grad_fn()->next_edges[0].get() == nullptr);
}

TEST_CASE("SumBackward broadcasts scalar grad uniformly to input shape") {
    const tensor::Tensor<float> x({2, 3}, std::vector<float>{1.f, 2.f, 3.f, 4.f, 5.f, 6.f}, true);
    const auto result = x.sum();

    const tensor::Tensor<float> propagated({}, std::vector<float>{2.f}, false);
    const auto grads = result.grad_fn()->apply(propagated);

    REQUIRE(grads.size() == 1);
    REQUIRE(grads[0].shape() == (std::vector<int64_t>{2, 3}));
    for (const float g : grads[0].data())
        CHECK(g == doctest::Approx(2.f));
}

// ─── sum(dim) ─────────────────────────────────────────────────────────────────

TEST_CASE("sum(dim=0) on 2-D tensor reduces rows") {
    // [[1, 2, 3], [4, 5, 6]] -> [5, 7, 9]
    const tensor::Tensor<float> x({2, 3}, std::vector<float>{1.f, 2.f, 3.f, 4.f, 5.f, 6.f}, false);
    const auto result = x.sum(0);

    CHECK(result.shape() == std::vector<int64_t>{3});
    CHECK(result.data()[0] == doctest::Approx(5.f));
    CHECK(result.data()[1] == doctest::Approx(7.f));
    CHECK(result.data()[2] == doctest::Approx(9.f));
}

TEST_CASE("sum(dim=1) on 2-D tensor reduces columns") {
    // [[1, 2, 3], [4, 5, 6]] -> [6, 15]
    const tensor::Tensor<float> x({2, 3}, std::vector<float>{1.f, 2.f, 3.f, 4.f, 5.f, 6.f}, false);
    const auto result = x.sum(1);

    CHECK(result.shape() == std::vector<int64_t>{2});
    CHECK(result.data()[0] == doctest::Approx(6.f));
    CHECK(result.data()[1] == doctest::Approx(15.f));
}

TEST_CASE("sum(dim) supports negative dimension index") {
    // Same as sum(dim=1) for a rank-2 tensor
    const tensor::Tensor<float> x({2, 3}, std::vector<float>{1.f, 2.f, 3.f, 4.f, 5.f, 6.f}, false);
    const auto result = x.sum(-1);

    CHECK(result.shape() == std::vector<int64_t>{2});
    CHECK(result.data()[0] == doctest::Approx(6.f));
    CHECK(result.data()[1] == doctest::Approx(15.f));
}

TEST_CASE("sum(dim=1) on 3-D tensor reduces middle axis") {
    // shape [2, 3, 2]: reduce dim 1 -> [2, 2]
    // data: [[[1,2],[3,4],[5,6]], [[7,8],[9,10],[11,12]]]
    const tensor::Tensor<float> x(
        {2, 3, 2},
        std::vector<float>{1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f},
        false);
    const auto result = x.sum(1);

    CHECK(result.shape() == (std::vector<int64_t>{2, 2}));
    // row 0: (1+3+5)=9, (2+4+6)=12
    CHECK(result.data()[0] == doctest::Approx(9.f));
    CHECK(result.data()[1] == doctest::Approx(12.f));
    // row 1: (7+9+11)=27, (8+10+12)=30
    CHECK(result.data()[2] == doctest::Approx(27.f));
    CHECK(result.data()[3] == doctest::Approx(30.f));
}

TEST_CASE("sum(dim) on rank-1 tensor produces scalar shape") {
    const tensor::Tensor<float> x({4}, std::vector<float>{1.f, 2.f, 3.f, 4.f}, false);
    const auto result = x.sum(0);

    CHECK(result.shape() == std::vector<int64_t>{});
    CHECK(result.data()[0] == doctest::Approx(10.f));
}

TEST_CASE("sum(dim) throws on scalar tensor") {
    const tensor::Tensor<float> x({}, false);
    CHECK_THROWS_AS(x.sum(0), std::invalid_argument);
}

TEST_CASE("sum(dim) throws on out-of-range dimension") {
    const tensor::Tensor<float> x({2, 3}, false);
    CHECK_THROWS_AS(x.sum(2),  std::out_of_range);
    CHECK_THROWS_AS(x.sum(-3), std::out_of_range);
}

TEST_CASE("SumDimBackward expands grad back along reduced dimension") {
    // [[1,2,3],[4,5,6]], sum(dim=0) -> [5,7,9]
    // backward: each input row gets the output grad
    const tensor::Tensor<float> x({2, 3}, std::vector<float>{1.f, 2.f, 3.f, 4.f, 5.f, 6.f}, true);
    const auto result = x.sum(0);

    REQUIRE(result.grad_fn().get() != nullptr);

    const tensor::Tensor<float> propagated({3}, std::vector<float>{1.f, 2.f, 3.f}, false);
    const auto grads = result.grad_fn()->apply(propagated);

    REQUIRE(grads.size() == 1);
    REQUIRE(grads[0].shape() == (std::vector<int64_t>{2, 3}));
    // Both rows should receive the same propagated grad
    CHECK(grads[0].data()[0] == doctest::Approx(1.f));
    CHECK(grads[0].data()[1] == doctest::Approx(2.f));
    CHECK(grads[0].data()[2] == doctest::Approx(3.f));
    CHECK(grads[0].data()[3] == doctest::Approx(1.f));
    CHECK(grads[0].data()[4] == doctest::Approx(2.f));
    CHECK(grads[0].data()[5] == doctest::Approx(3.f));
}

// ─── mean() global ────────────────────────────────────────────────────────────

TEST_CASE("mean() reduces 1-D tensor to scalar") {
    const tensor::Tensor<float> x({4}, std::vector<float>{2.f, 4.f, 6.f, 8.f}, false);
    const auto result = x.mean();

    CHECK(result.shape() == std::vector<int64_t>{});
    CHECK(result.data()[0] == doctest::Approx(5.f));
}

TEST_CASE("mean() reduces 2-D tensor to scalar") {
    // [[1,2],[3,4]] -> 2.5
    const tensor::Tensor<double> x({2, 2}, std::vector<double>{1.0, 2.0, 3.0, 4.0}, false);
    CHECK(x.mean().data()[0] == doctest::Approx(2.5));
}

TEST_CASE("mean() throws on empty tensor") {
    const tensor::Tensor<float> x({0}, false);
    CHECK_THROWS_AS(x.mean(), std::invalid_argument);
}

TEST_CASE("mean() does not track grad when requires_grad=false") {
    const tensor::Tensor<float> x({3}, std::vector<float>{1.f, 2.f, 3.f}, false);
    CHECK_FALSE(x.mean().requires_grad());
}

TEST_CASE("MeanBackward distributes grad evenly as 1/numel") {
    const tensor::Tensor<float> x({2, 3}, std::vector<float>{1.f, 2.f, 3.f, 4.f, 5.f, 6.f}, true);
    const auto result = x.mean();

    REQUIRE(result.grad_fn().get() != nullptr);

    const tensor::Tensor<float> propagated({}, std::vector<float>{6.f}, false);
    const auto grads = result.grad_fn()->apply(propagated);

    REQUIRE(grads.size() == 1);
    REQUIRE(grads[0].shape() == (std::vector<int64_t>{2, 3}));
    for (const float g : grads[0].data())
        CHECK(g == doctest::Approx(1.f)); // 6 / 6 = 1
}

// ─── mean(dim) ────────────────────────────────────────────────────────────────

TEST_CASE("mean(dim=0) on 2-D tensor averages rows") {
    // [[1,2,3],[3,4,5]] -> [2, 3, 4]
    const tensor::Tensor<float> x({2, 3}, std::vector<float>{1.f, 2.f, 3.f, 3.f, 4.f, 5.f}, false);
    const auto result = x.mean(0);

    CHECK(result.shape() == std::vector<int64_t>{3});
    CHECK(result.data()[0] == doctest::Approx(2.f));
    CHECK(result.data()[1] == doctest::Approx(3.f));
    CHECK(result.data()[2] == doctest::Approx(4.f));
}

TEST_CASE("mean(dim=1) on 2-D tensor averages columns") {
    // [[1,2,3],[4,5,6]] -> [2, 5]
    const tensor::Tensor<float> x({2, 3}, std::vector<float>{1.f, 2.f, 3.f, 4.f, 5.f, 6.f}, false);
    const auto result = x.mean(1);

    CHECK(result.shape() == std::vector<int64_t>{2});
    CHECK(result.data()[0] == doctest::Approx(2.f));
    CHECK(result.data()[1] == doctest::Approx(5.f));
}

TEST_CASE("mean(dim) supports negative dimension index") {
    const tensor::Tensor<float> x({2, 3}, std::vector<float>{1.f, 2.f, 3.f, 4.f, 5.f, 6.f}, false);
    const auto pos = x.mean(1);
    const auto neg = x.mean(-1);

    REQUIRE(pos.shape() == neg.shape());
    for (size_t i = 0; i < static_cast<size_t>(pos.numel()); ++i)
        CHECK(pos.data()[i] == doctest::Approx(neg.data()[i]));
}

TEST_CASE("mean(dim) throws on scalar tensor") {
    const tensor::Tensor<float> x({}, false);
    CHECK_THROWS_AS(x.mean(0), std::invalid_argument);
}

TEST_CASE("mean(dim) throws on empty reduced dimension") {
    const tensor::Tensor<float> x({2, 0}, false);
    CHECK_THROWS_AS(x.mean(1), std::invalid_argument);
}

TEST_CASE("MeanDimBackward distributes grad as 1/dim_size along reduced axis") {
    // [[1,2,3],[4,5,6]], mean(dim=0) -> [2.5, 3.5, 4.5]
    // backward with grad [1,1,1]: each input element gets 1/2 = 0.5
    const tensor::Tensor<float> x({2, 3}, std::vector<float>{1.f, 2.f, 3.f, 4.f, 5.f, 6.f}, true);
    const auto result = x.mean(0);

    REQUIRE(result.grad_fn().get() != nullptr);

    const tensor::Tensor<float> propagated({3}, std::vector<float>{1.f, 1.f, 1.f}, false);
    const auto grads = result.grad_fn()->apply(propagated);

    REQUIRE(grads.size() == 1);
    REQUIRE(grads[0].shape() == (std::vector<int64_t>{2, 3}));
    for (const float g : grads[0].data())
        CHECK(g == doctest::Approx(0.5f));
}

// ─── max() ────────────────────────────────────────────────────────────────────

TEST_CASE("max() returns maximum element as scalar") {
    const tensor::Tensor<float> x({4}, std::vector<float>{3.f, 1.f, 4.f, 2.f}, false);
    const auto result = x.max();

    CHECK(result.shape() == std::vector<int64_t>{});
    CHECK(result.data()[0] == doctest::Approx(4.f));
}

TEST_CASE("max() works on 2-D tensor") {
    const tensor::Tensor<double> x({2, 3}, std::vector<double>{1.0, 5.0, 3.0, 2.0, 4.0, 0.0}, false);
    CHECK(x.max().data()[0] == doctest::Approx(5.0));
}

TEST_CASE("max() throws on empty tensor") {
    const tensor::Tensor<float> x({0}, false);
    CHECK_THROWS_AS(x.max(), std::invalid_argument);
}

TEST_CASE("MaxBackward passes grad only to the argmax position") {
    // max is at index 2 (value 9)
    const tensor::Tensor<float> x({4}, std::vector<float>{1.f, 3.f, 9.f, 2.f}, true);
    const auto result = x.max();

    REQUIRE(result.grad_fn().get() != nullptr);

    const tensor::Tensor<float> propagated({}, std::vector<float>{5.f}, false);
    const auto grads = result.grad_fn()->apply(propagated);

    REQUIRE(grads.size() == 1);
    REQUIRE(grads[0].shape() == std::vector<int64_t>{4});
    CHECK(grads[0].data()[0] == doctest::Approx(0.f));
    CHECK(grads[0].data()[1] == doctest::Approx(0.f));
    CHECK(grads[0].data()[2] == doctest::Approx(5.f));
    CHECK(grads[0].data()[3] == doctest::Approx(0.f));
}

TEST_CASE("MaxBackward on tie picks first occurrence") {
    const tensor::Tensor<float> x({4}, std::vector<float>{5.f, 5.f, 5.f, 5.f}, true);
    const auto result = x.max();
    const tensor::Tensor<float> propagated({}, std::vector<float>{1.f}, false);
    const auto grads = result.grad_fn()->apply(propagated);

    // Only index 0 gets the gradient
    CHECK(grads[0].data()[0] == doctest::Approx(1.f));
    CHECK(grads[0].data()[1] == doctest::Approx(0.f));
    CHECK(grads[0].data()[2] == doctest::Approx(0.f));
    CHECK(grads[0].data()[3] == doctest::Approx(0.f));
}

// ─── min() ────────────────────────────────────────────────────────────────────

TEST_CASE("min() returns minimum element as scalar") {
    const tensor::Tensor<float> x({4}, std::vector<float>{3.f, 1.f, 4.f, 2.f}, false);
    const auto result = x.min();

    CHECK(result.shape() == std::vector<int64_t>{});
    CHECK(result.data()[0] == doctest::Approx(1.f));
}

TEST_CASE("min() works on 2-D tensor") {
    const tensor::Tensor<double> x({2, 3}, std::vector<double>{1.0, 5.0, 3.0, 2.0, -4.0, 0.0}, false);
    CHECK(x.min().data()[0] == doctest::Approx(-4.0));
}

TEST_CASE("min() throws on empty tensor") {
    const tensor::Tensor<float> x({0}, false);
    CHECK_THROWS_AS(x.min(), std::invalid_argument);
}

TEST_CASE("MinBackward passes grad only to the argmin position") {
    // min is at index 1 (value -2)
    const tensor::Tensor<float> x({4}, std::vector<float>{3.f, -2.f, 5.f, 1.f}, true);
    const auto result = x.min();

    REQUIRE(result.grad_fn().get() != nullptr);

    const tensor::Tensor<float> propagated({}, std::vector<float>{3.f}, false);
    const auto grads = result.grad_fn()->apply(propagated);

    REQUIRE(grads.size() == 1);
    REQUIRE(grads[0].shape() == std::vector<int64_t>{4});
    CHECK(grads[0].data()[0] == doctest::Approx(0.f));
    CHECK(grads[0].data()[1] == doctest::Approx(3.f));
    CHECK(grads[0].data()[2] == doctest::Approx(0.f));
    CHECK(grads[0].data()[3] == doctest::Approx(0.f));
}

TEST_CASE("MinBackward on tie picks first occurrence") {
    const tensor::Tensor<float> x({4}, std::vector<float>{2.f, 2.f, 2.f, 2.f}, true);
    const auto result = x.min();
    const tensor::Tensor<float> propagated({}, std::vector<float>{1.f}, false);
    const auto grads = result.grad_fn()->apply(propagated);
    CHECK(grads[0].data()[0] == doctest::Approx(1.f));
    CHECK(grads[0].data()[1] == doctest::Approx(0.f));
    CHECK(grads[0].data()[2] == doctest::Approx(0.f));
    CHECK(grads[0].data()[3] == doctest::Approx(0.f));
}

// ─── max(dim) ─────────────────────────────────────────────────────────────────

TEST_CASE("max(dim=0) on 2-D tensor reduces rows") {
    const tensor::Tensor<float> x({2, 3}, std::vector<float>{1.f, 5.f, 3.f, 4.f, 2.f, 6.f}, false);
    const auto result = x.max(0);

    CHECK(result.shape() == std::vector<int64_t>{3});
    CHECK(result.data()[0] == doctest::Approx(4.f));
    CHECK(result.data()[1] == doctest::Approx(5.f));
    CHECK(result.data()[2] == doctest::Approx(6.f));
}

TEST_CASE("max(dim=1) on 2-D tensor reduces columns") {
    const tensor::Tensor<float> x({2, 3}, std::vector<float>{1.f, 5.f, 3.f, 4.f, 2.f, 6.f}, false);
    const auto result = x.max(1);

    CHECK(result.shape() == std::vector<int64_t>{2});
    CHECK(result.data()[0] == doctest::Approx(5.f));
    CHECK(result.data()[1] == doctest::Approx(6.f));
}

TEST_CASE("max(dim) supports negative dimension index") {
    const tensor::Tensor<float> x({2, 3}, std::vector<float>{1.f, 5.f, 3.f, 4.f, 2.f, 6.f}, false);
    const auto pos = x.max(1);
    const auto neg = x.max(-1);
    REQUIRE(pos.shape() == neg.shape());
    for (size_t i = 0; i < static_cast<size_t>(pos.numel()); ++i)
        CHECK(pos.data()[i] == doctest::Approx(neg.data()[i]));
}

TEST_CASE("max(dim=1) on 3-D tensor reduces middle axis") {
    const tensor::Tensor<float> x(
        {2, 3, 2},
        std::vector<float>{1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f},
        false);
    const auto result = x.max(1);

    CHECK(result.shape() == (std::vector<int64_t>{2, 2}));
    CHECK(result.data()[0] == doctest::Approx(5.f));
    CHECK(result.data()[1] == doctest::Approx(6.f));
    CHECK(result.data()[2] == doctest::Approx(11.f));
    CHECK(result.data()[3] == doctest::Approx(12.f));
}

TEST_CASE("max(dim) on rank-1 tensor produces scalar shape") {
    const tensor::Tensor<float> x({4}, std::vector<float>{3.f, 1.f, 4.f, 2.f}, false);
    const auto result = x.max(0);
    CHECK(result.shape() == std::vector<int64_t>{});
    CHECK(result.data()[0] == doctest::Approx(4.f));
}

TEST_CASE("max(dim) throws on scalar tensor") {
    const tensor::Tensor<float> x({}, false);
    CHECK_THROWS_AS(x.max(0), std::invalid_argument);
}

TEST_CASE("max(dim) throws on out-of-range dimension") {
    const tensor::Tensor<float> x({2, 3}, false);
    CHECK_THROWS_AS(x.max(2), std::out_of_range);
    CHECK_THROWS_AS(x.max(-3), std::out_of_range);
}

TEST_CASE("max(dim) throws on empty reduced dimension") {
    const tensor::Tensor<float> x({2, 0}, false);
    CHECK_THROWS_AS(x.max(1), std::invalid_argument);
}

TEST_CASE("MaxDimBackward scatters grad to the argmax along dim") {
    const tensor::Tensor<float> x({2, 3}, std::vector<float>{1.f, 5.f, 3.f, 4.f, 2.f, 6.f}, true);
    const auto result = x.max(0);
    REQUIRE(result.grad_fn().get() != nullptr);

    const tensor::Tensor<float> propagated({3}, std::vector<float>{10.f, 20.f, 30.f}, false);
    const auto grads = result.grad_fn()->apply(propagated);

    REQUIRE(grads.size() == 1);
    REQUIRE(grads[0].shape() == (std::vector<int64_t>{2, 3}));
    CHECK(grads[0].data()[0] == doctest::Approx(0.f));
    CHECK(grads[0].data()[1] == doctest::Approx(20.f));
    CHECK(grads[0].data()[2] == doctest::Approx(0.f));
    CHECK(grads[0].data()[3] == doctest::Approx(10.f));
    CHECK(grads[0].data()[4] == doctest::Approx(0.f));
    CHECK(grads[0].data()[5] == doctest::Approx(30.f));
}

TEST_CASE("MaxDimBackward on tie picks first occurrence along dim") {
    const tensor::Tensor<float> x({2, 3}, std::vector<float>{5.f, 1.f, 3.f, 5.f, 2.f, 3.f}, true);
    const auto result = x.max(0);
    const tensor::Tensor<float> propagated({3}, std::vector<float>{1.f, 1.f, 1.f}, false);
    const auto grads = result.grad_fn()->apply(propagated);

    CHECK(grads[0].data()[0] == doctest::Approx(1.f));
    CHECK(grads[0].data()[1] == doctest::Approx(0.f));
    CHECK(grads[0].data()[2] == doctest::Approx(1.f));
    CHECK(grads[0].data()[3] == doctest::Approx(0.f));
    CHECK(grads[0].data()[4] == doctest::Approx(1.f));
    CHECK(grads[0].data()[5] == doctest::Approx(0.f));
}

// ─── min(dim) ─────────────────────────────────────────────────────────────────

TEST_CASE("min(dim=0) on 2-D tensor reduces rows") {
    const tensor::Tensor<float> x({2, 3}, std::vector<float>{1.f, 5.f, 3.f, 4.f, 2.f, 6.f}, false);
    const auto result = x.min(0);

    CHECK(result.shape() == std::vector<int64_t>{3});
    CHECK(result.data()[0] == doctest::Approx(1.f));
    CHECK(result.data()[1] == doctest::Approx(2.f));
    CHECK(result.data()[2] == doctest::Approx(3.f));
}

TEST_CASE("min(dim=1) on 2-D tensor reduces columns") {
    const tensor::Tensor<float> x({2, 3}, std::vector<float>{1.f, 5.f, 3.f, 4.f, 2.f, 6.f}, false);
    const auto result = x.min(1);

    CHECK(result.shape() == std::vector<int64_t>{2});
    CHECK(result.data()[0] == doctest::Approx(1.f));
    CHECK(result.data()[1] == doctest::Approx(2.f));
}

TEST_CASE("min(dim) supports negative dimension index") {
    const tensor::Tensor<float> x({2, 3}, std::vector<float>{1.f, 5.f, 3.f, 4.f, 2.f, 6.f}, false);
    CHECK(x.min(-1).data()[0] == doctest::Approx(x.min(1).data()[0]));
    CHECK(x.min(-1).data()[1] == doctest::Approx(x.min(1).data()[1]));
}

TEST_CASE("min(dim=1) on 3-D tensor reduces middle axis") {
    const tensor::Tensor<float> x(
        {2, 3, 2},
        std::vector<float>{1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f},
        false);
    const auto result = x.min(1);

    CHECK(result.shape() == (std::vector<int64_t>{2, 2}));
    CHECK(result.data()[0] == doctest::Approx(1.f));
    CHECK(result.data()[1] == doctest::Approx(2.f));
    CHECK(result.data()[2] == doctest::Approx(7.f));
    CHECK(result.data()[3] == doctest::Approx(8.f));
}

TEST_CASE("min(dim) on rank-1 tensor produces scalar shape") {
    const tensor::Tensor<float> x({4}, std::vector<float>{3.f, 1.f, 4.f, 2.f}, false);
    const auto result = x.min(0);
    CHECK(result.shape() == std::vector<int64_t>{});
    CHECK(result.data()[0] == doctest::Approx(1.f));
}

TEST_CASE("min(dim) throws on scalar tensor") {
    const tensor::Tensor<float> x({}, false);
    CHECK_THROWS_AS(x.min(0), std::invalid_argument);
}

TEST_CASE("min(dim) throws on out-of-range dimension") {
    const tensor::Tensor<float> x({2, 3}, false);
    CHECK_THROWS_AS(x.min(2), std::out_of_range);
    CHECK_THROWS_AS(x.min(-3), std::out_of_range);
}

TEST_CASE("min(dim) throws on empty reduced dimension") {
    const tensor::Tensor<float> x({2, 0}, false);
    CHECK_THROWS_AS(x.min(1), std::invalid_argument);
}

TEST_CASE("MinDimBackward scatters grad to the argmin along dim") {
    const tensor::Tensor<float> x({2, 3}, std::vector<float>{1.f, 5.f, 3.f, 4.f, 2.f, 6.f}, true);
    const auto result = x.min(0);
    REQUIRE(result.grad_fn().get() != nullptr);

    const tensor::Tensor<float> propagated({3}, std::vector<float>{10.f, 20.f, 30.f}, false);
    const auto grads = result.grad_fn()->apply(propagated);

    REQUIRE(grads.size() == 1);
    REQUIRE(grads[0].shape() == (std::vector<int64_t>{2, 3}));
    CHECK(grads[0].data()[0] == doctest::Approx(10.f));
    CHECK(grads[0].data()[1] == doctest::Approx(0.f));
    CHECK(grads[0].data()[2] == doctest::Approx(30.f));
    CHECK(grads[0].data()[3] == doctest::Approx(0.f));
    CHECK(grads[0].data()[4] == doctest::Approx(20.f));
    CHECK(grads[0].data()[5] == doctest::Approx(0.f));
}

TEST_CASE("MinDimBackward on tie picks first occurrence along dim") {
    const tensor::Tensor<float> x({2, 3}, std::vector<float>{5.f, 2.f, 3.f, 1.f, 2.f, 3.f}, true);
    const auto result = x.min(0);
    const tensor::Tensor<float> propagated({3}, std::vector<float>{1.f, 1.f, 1.f}, false);
    const auto grads = result.grad_fn()->apply(propagated);

    CHECK(grads[0].data()[0] == doctest::Approx(0.f));
    CHECK(grads[0].data()[1] == doctest::Approx(1.f));
    CHECK(grads[0].data()[2] == doctest::Approx(1.f));
    CHECK(grads[0].data()[3] == doctest::Approx(1.f));
    CHECK(grads[0].data()[4] == doctest::Approx(0.f));
    CHECK(grads[0].data()[5] == doctest::Approx(0.f));
}

// ─── grad node wiring ─────────────────────────────────────────────────────────

TEST_CASE("reduction ops produce no grad_fn when all inputs are no-grad") {
    const tensor::Tensor<float> x({3}, std::vector<float>{1.f, 2.f, 3.f}, false);

    CHECK(x.sum().grad_fn()    .get() == nullptr);
    CHECK(x.sum(0).grad_fn()   .get() == nullptr);
    CHECK(x.mean().grad_fn()   .get() == nullptr);
    CHECK(x.mean(0).grad_fn()  .get() == nullptr);
    CHECK(x.max().grad_fn()    .get() == nullptr);
    CHECK(x.max(0).grad_fn()   .get() == nullptr);
    CHECK(x.min().grad_fn()    .get() == nullptr);
    CHECK(x.min(0).grad_fn()   .get() == nullptr);
}

TEST_CASE("softmax along dim=1 normalizes each row") {
    const tensor::Tensor<float> x({2, 3}, std::vector<float>{1.f, 2.f, 3.f, 1.f, 1.f, 1.f}, false);
    const auto s = x.softmax(1);

    REQUIRE(s.shape() == std::vector<int64_t>{2, 3});
    const float row0 = std::exp(1.f) + std::exp(2.f) + std::exp(3.f);
    CHECK(s.data()[0] == doctest::Approx(std::exp(1.f) / row0));
    CHECK(s.data()[1] == doctest::Approx(std::exp(2.f) / row0));
    CHECK(s.data()[2] == doctest::Approx(std::exp(3.f) / row0));
    CHECK(s.data()[3] == doctest::Approx(1.f / 3.f));
    CHECK(s.data()[4] == doctest::Approx(1.f / 3.f));
    CHECK(s.data()[5] == doctest::Approx(1.f / 3.f));
}

TEST_CASE("softmax along a non-last axis normalizes each inner slab") {
    // shape [2, 3, 2]; dim=1 → two independent 3-vectors per outer, inner=2
    const tensor::Tensor<float> x(
        {2, 3, 2},
        std::vector<float>{1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f},
        false);
    const auto s = x.softmax(1);
    REQUIRE(s.shape() == (std::vector<int64_t>{2, 3, 2}));

    const float e1 = std::exp(1.f);
    const float e3 = std::exp(3.f);
    const float e5 = std::exp(5.f);
    const float den0 = e1 + e3 + e5;
    CHECK(s.data()[0] == doctest::Approx(e1 / den0));
    CHECK(s.data()[2] == doctest::Approx(e3 / den0));
    CHECK(s.data()[4] == doctest::Approx(e5 / den0));

    const float e2 = std::exp(2.f);
    const float e4 = std::exp(4.f);
    const float e6 = std::exp(6.f);
    const float den1 = e2 + e4 + e6;
    CHECK(s.data()[1] == doctest::Approx(e2 / den1));
    CHECK(s.data()[3] == doctest::Approx(e4 / den1));
    CHECK(s.data()[5] == doctest::Approx(e6 / den1));

    for (size_t i = 6; i < 12; ++i)
        CHECK(s.data()[i] == doctest::Approx(1.f / 3.f));
}

TEST_CASE("SoftmaxBackward matches s * (g - sum(g * s))") {
    const tensor::Tensor<float> x({2}, std::vector<float>{1.f, 2.f}, true);
    const auto s = x.softmax(0);
    REQUIRE(s.grad_fn().get() != nullptr);

    const tensor::Tensor<float> g({2}, std::vector<float>{1.f, 0.f}, false);
    const auto grads = s.grad_fn()->apply(g);

    REQUIRE(grads.size() == 1);
    const float s0 = s.data()[0];
    const float s1 = s.data()[1];
    // dx_i = s_i * (g_i - s_0) when g = [1, 0]
    CHECK(grads[0].data()[0] == doctest::Approx(s0 * (1.f - s0)));
    CHECK(grads[0].data()[1] == doctest::Approx(s1 * (0.f - s0)));
}

TEST_CASE("reduction ops wire next_edge to input grad_fn") {
    // Build a simple chain: leaf -> add -> sum, so leaf has no grad_fn
    // but the add result does. Verify sum's next_edge points at add's node.
    const tensor::Tensor<float> a({3}, std::vector<float>{1.f, 2.f, 3.f}, true);
    const tensor::Tensor<float> b({3}, std::vector<float>{1.f, 1.f, 1.f}, true);
    const auto mid    = a.add(b);   // has grad_fn
    const auto result = mid.sum();

    REQUIRE(result.grad_fn().get() != nullptr);
    REQUIRE(result.grad_fn()->next_edges.size() == 1);
    CHECK(result.grad_fn()->next_edges[0].get() == mid.grad_fn().get());
}

TEST_CASE("sum of a transpose equals the sum of logical elements") {
    const tensor::Tensor<float> t({2, 3}, std::vector<float>{1.f, 2.f, 3.f, 4.f, 5.f, 6.f}, false);
    CHECK(t.transpose().sum().data()[0] == doctest::Approx(21.f));
}

TEST_CASE("sum of a broadcast counts repeated logical elements") {
    const tensor::Tensor<float> t({1, 3}, std::vector<float>{1.f, 2.f, 3.f}, false);
    CHECK(t.broadcast_to({2, 3}).sum().data()[0] == doctest::Approx(12.f));
}

TEST_CASE("sum of a narrow is only the slice") {
    const tensor::Tensor<float> t({2, 3}, std::vector<float>{1.f, 2.f, 3.f, 4.f, 5.f, 6.f}, false);
    CHECK(t.narrow(1, 1, 2).sum().data()[0] == doctest::Approx(16.f));
}
