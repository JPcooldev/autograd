/*
 * Add after shape ops must pack logical order, not walk storage order.
 *
 * - add after transpose
 * - add after each of transpose, reshape, view, flatten, squeeze, unsqueeze,
 *   broadcast_to, contiguous, narrow, cat
 * - add after chained shape ops
 */

#include <cstdint>
#include <stdexcept>
#include <vector>

#include "../doctest/doctest.h"

#include "../../src/autograd/autograd.h"

// Kernels such as add walk data()[i] densely. Shape ops often return views
// (shared storage, different strides/offset). Without a pack at the kernel
// door, storage-order iteration disagrees with logical layout.
//
// Running example:
//   A     [[1, 2, 3], [4, 5, 6]]     shape {2,3}  storage [1,2,3,4,5,6]  strides {3,1}
//   B     [[2, 8], [4, 10], [6, 12]] shape {3,2}  storage [2,8,4,10,6,12] strides {2,1}
//   A.T   [[1, 4], [2, 5], [3, 6]]   shape {3,2}  same storage as A       strides {1,3}
//   A.T+B [[3, 12], [6, 15], [9, 18]] packed [3,12,6,15,9,18]
// Naive storage-order add would yield [3,10,7,14,9,18].

namespace {

/**
 * Assert `lhs.add(rhs)` packs logical order into a dense buffer matching `expected`.
 *
 * @param lhs Left addend.
 * @param rhs Right addend; must match `lhs.shape()`.
 * @param expected Packed row-major values of the sum.
 */
void check_packed_add(
    const tensor::Tensor<int32_t>& lhs,
    const tensor::Tensor<int32_t>& rhs,
    const std::vector<int32_t>& expected) {
    REQUIRE(lhs.shape() == rhs.shape());
    const auto out = lhs.add(rhs);
    CHECK(out.shape() == lhs.shape());
    CHECK(out.is_contiguous());
    CHECK(out.offset() == 0);
    CHECK(out.data() == expected);
}

/**
 * Build a no-grad ones tensor with the same shape as `t`.
 *
 * @param t Shape source.
 * @return `ones(t.shape(), false)`.
 */
tensor::Tensor<int32_t> ones_like(const tensor::Tensor<int32_t>& t) {
    return tensor::Tensor<int32_t>::ones(t.shape(), false);
}

} // namespace

TEST_CASE("add after transpose uses logical order, not storage order") {
    const tensor::Tensor<int32_t> A({2, 3}, std::vector<int32_t>{1, 2, 3, 4, 5, 6}, false);
    const tensor::Tensor<int32_t> B({3, 2}, std::vector<int32_t>{2, 8, 4, 10, 6, 12}, false);
    const auto At = A.transpose();

    CHECK(At.shape() == std::vector<int64_t>{3, 2});
    CHECK(At.strides() == std::vector<int64_t>{1, 3});
    CHECK(At.is_view());
    CHECK_FALSE(At.is_contiguous());
    CHECK(At.data() == A.data());

    const std::vector<int32_t> expected{3, 12, 6, 15, 9, 18};
    check_packed_add(At, B, expected);
    check_packed_add(B, At, expected);
    CHECK(A.data() == std::vector<int32_t>{1, 2, 3, 4, 5, 6});
}

TEST_CASE("add after each shape op packs logical order") {
    const tensor::Tensor<int32_t> A({2, 3}, std::vector<int32_t>{1, 2, 3, 4, 5, 6}, false);
    const tensor::Tensor<int32_t> B({3, 2}, std::vector<int32_t>{2, 8, 4, 10, 6, 12}, false);

    SUBCASE("reshape of a contiguous tensor keeps storage order") {
        // reshape is not a permute: [[1,2],[3,4],[5,6]] + B
        check_packed_add(A.reshape({3, 2}), B, {3, 10, 7, 14, 11, 18});
    }

    SUBCASE("view of a contiguous tensor matches reshape") {
        check_packed_add(A.view({3, 2}), B, {3, 10, 7, 14, 11, 18});
    }

    SUBCASE("view of a transpose throws rather than packing") {
        CHECK_THROWS_AS(A.transpose().view({3, 2}), std::invalid_argument);
    }

    SUBCASE("flatten of a contiguous tensor") {
        const tensor::Tensor<int32_t> v({6}, std::vector<int32_t>{10, 20, 30, 40, 50, 60}, false);
        check_packed_add(A.flatten(), v, {11, 22, 33, 44, 55, 66});
    }

    SUBCASE("flatten of a transpose packs first") {
        const tensor::Tensor<int32_t> v({6}, std::vector<int32_t>{10, 20, 30, 40, 50, 60}, false);
        check_packed_add(A.transpose().flatten(), v, {11, 24, 32, 45, 53, 66});
    }

    SUBCASE("unsqueeze of a contiguous tensor") {
        check_packed_add(A.unsqueeze(0), ones_like(A.unsqueeze(0)), {2, 3, 4, 5, 6, 7});
    }

    SUBCASE("unsqueeze of a transpose") {
        const auto u = A.transpose().unsqueeze(0);
        CHECK(u.shape() == std::vector<int64_t>{1, 3, 2});
        CHECK_FALSE(u.is_contiguous());
        check_packed_add(u, ones_like(u), {2, 5, 3, 6, 4, 7});
    }

    SUBCASE("squeeze restores axes without changing logical values") {
        check_packed_add(A.unsqueeze(0).squeeze(), ones_like(A), {2, 3, 4, 5, 6, 7});
        check_packed_add(A.transpose().unsqueeze(0).squeeze(0), B, {3, 12, 6, 15, 9, 18});
    }

    SUBCASE("broadcast_to repeats with stride 0") {
        const tensor::Tensor<int32_t> col({3, 1}, std::vector<int32_t>{1, 2, 3}, false);
        const auto expanded = col.broadcast_to({3, 2});
        CHECK(expanded.strides()[1] == 0);
        CHECK(expanded.is_view());
        check_packed_add(expanded, B, {3, 9, 6, 12, 9, 15});
    }

    SUBCASE("narrow with non-contiguous row stride") {
        // columns 1..2: [[2,3],[5,6]]; row stride stays 3
        const auto n = A.narrow(1, 1, 2);
        CHECK(n.shape() == std::vector<int64_t>{2, 2});
        CHECK(n.is_view());
        CHECK_FALSE(n.is_contiguous());
        check_packed_add(n, ones_like(n), {3, 4, 6, 7});
    }

    SUBCASE("narrow with offset != 0 on an otherwise contiguous slice") {
        // second row: [4,5,6]; contiguous strides, but offset 3
        const auto n = A.narrow(0, 1, 1);
        CHECK(n.shape() == std::vector<int64_t>{1, 3});
        CHECK(n.is_contiguous());
        CHECK(n.offset() != 0);
        check_packed_add(n, ones_like(n), {5, 6, 7});
    }

    SUBCASE("contiguous of a transpose is already packed") {
        check_packed_add(A.transpose().contiguous(), B, {3, 12, 6, 15, 9, 18});
    }

    SUBCASE("cat of transposes copies logical order") {
        const auto c = ops::cat({A.transpose(), A.transpose()}, 1);
        CHECK(c.shape() == std::vector<int64_t>{3, 4});
        check_packed_add(c, ones_like(c), {2, 5, 2, 5, 3, 6, 3, 6, 4, 7, 4, 7});
    }
}

TEST_CASE("add after chained shape ops packs logical order") {
    const tensor::Tensor<int32_t> A({2, 3}, std::vector<int32_t>{1, 2, 3, 4, 5, 6}, false);
    const tensor::Tensor<int32_t> B({3, 2}, std::vector<int32_t>{2, 8, 4, 10, 6, 12}, false);

    SUBCASE("reshape then transpose") {
        // [[1,2],[3,4],[5,6]]^T → [[1,3,5],[2,4,6]]
        const auto t = A.reshape({3, 2}).transpose();
        check_packed_add(t, ones_like(t), {2, 4, 6, 3, 5, 7});
    }

    SUBCASE("view then transpose") {
        const auto t = A.view({3, 2}).transpose();
        check_packed_add(t, ones_like(t), {2, 4, 6, 3, 5, 7});
    }

    SUBCASE("transpose then reshape packs before the new shape") {
        // A.T packed [1,4,2,5,3,6] viewed as {2,3}
        check_packed_add(A.transpose().reshape({2, 3}), ones_like(A), {2, 5, 3, 6, 4, 7});
    }

    SUBCASE("transpose then flatten") {
        const tensor::Tensor<int32_t> v({6}, std::vector<int32_t>{1, 1, 1, 1, 1, 1}, false);
        check_packed_add(A.transpose().flatten(), v, {2, 5, 3, 6, 4, 7});
    }

    SUBCASE("transpose then unsqueeze then squeeze is transpose") {
        check_packed_add(A.transpose().unsqueeze(-1).squeeze(-1), B, {3, 12, 6, 15, 9, 18});
    }

    SUBCASE("unsqueeze then broadcast_to") {
        const auto t = A.unsqueeze(-1).broadcast_to({2, 3, 2});
        check_packed_add(t, ones_like(t), {2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7});
    }

    SUBCASE("narrow then transpose") {
        // [[2,3],[5,6]]^T → [[2,5],[3,6]]
        const auto t = A.narrow(1, 1, 2).transpose();
        check_packed_add(t, ones_like(t), {3, 6, 4, 7});
    }

    SUBCASE("transpose then narrow") {
        // A.T rows 1..2: [[2,5],[3,6]]
        const auto t = A.transpose().narrow(0, 1, 2);
        check_packed_add(t, ones_like(t), {3, 6, 4, 7});
    }

    SUBCASE("broadcast_to then transpose") {
        const tensor::Tensor<int32_t> col({3, 1}, std::vector<int32_t>{1, 2, 3}, false);
        const auto t = col.broadcast_to({3, 2}).transpose();
        check_packed_add(t, ones_like(t), {2, 3, 4, 2, 3, 4});
    }

    SUBCASE("squeeze then transpose") {
        check_packed_add(A.unsqueeze(0).squeeze(0).transpose(), B, {3, 12, 6, 15, 9, 18});
    }

    SUBCASE("double transpose is the original layout") {
        check_packed_add(A.transpose().transpose(), ones_like(A), {2, 3, 4, 5, 6, 7});
    }

    SUBCASE("both operands are non-contiguous views") {
        const tensor::Tensor<int32_t> C({2, 3}, std::vector<int32_t>{2, 4, 6, 8, 10, 12}, false);
        check_packed_add(A.transpose(), C.transpose(), {3, 12, 6, 15, 9, 18});
    }
}
