/*
 * Math kernels on views use logical order, including a few backward cases.
 *
 * - add / multiply / neg / exp / relu after transpose
 * - mean and max after transpose
 * - sum(dim) after transpose
 * - mean / max / min / softmax after transpose
 * - add after broadcast_to; relu after narrow
 * - l1_loss after transpose
 * - add after transpose backprops onto the leaf
 * - multiply after transpose backprops the other operand
 */

#include <cmath>
#include <vector>

#include "../doctest/doctest.h"

#include "../../src/autograd/autograd.h"
#include "../../src/ops/loss_ops.h"

// Owner [2, 3]:
//   1 2 3
//   4 5 6
// transpose [3, 2] logical order (packed): 1, 4, 2, 5, 3, 6

TEST_CASE("add after transpose uses logical order") {
    const tensor::Tensor<float> t({2, 3}, std::vector<float>{1.f, 2.f, 3.f, 4.f, 5.f, 6.f}, false);
    const tensor::Tensor<float> ones({3, 2}, std::vector<float>(6, 1.f), false);
    const auto r = t.transpose().add(ones);
    CHECK(r.is_contiguous());
    CHECK(r.data() == std::vector<float>{2.f, 5.f, 3.f, 6.f, 4.f, 7.f});
}

TEST_CASE("multiply after transpose uses logical order") {
    const tensor::Tensor<float> t({2, 3}, std::vector<float>{1.f, 2.f, 3.f, 4.f, 5.f, 6.f}, false);
    const tensor::Tensor<float> scale({3, 2}, std::vector<float>{10.f, 10.f, 10.f, 10.f, 10.f, 10.f}, false);
    const auto r = t.transpose().multiply(scale);
    CHECK(r.data() == std::vector<float>{10.f, 40.f, 20.f, 50.f, 30.f, 60.f});
}

TEST_CASE("neg and exp after transpose use logical order") {
    const tensor::Tensor<float> t({2, 3}, std::vector<float>{1.f, 2.f, 3.f, 4.f, 5.f, 6.f}, false);
    const auto n = t.transpose().neg();
    CHECK(n.data() == std::vector<float>{-1.f, -4.f, -2.f, -5.f, -3.f, -6.f});

    const auto e = t.transpose().exp();
    const std::vector<float> logical{1.f, 4.f, 2.f, 5.f, 3.f, 6.f};
    for (size_t i = 0; i < logical.size(); ++i)
        CHECK(e.data()[i] == doctest::Approx(std::exp(logical[i])));
}

TEST_CASE("relu after transpose zeros negatives in logical order") {
    const tensor::Tensor<float> t({2, 3}, std::vector<float>{1.f, -2.f, 3.f, 4.f, -5.f, 6.f}, false);
    const auto r = t.transpose().relu();
    CHECK(r.shape() == std::vector<int64_t>{3, 2});
    CHECK(r.data() == std::vector<float>{1.f, 4.f, 0.f, 0.f, 3.f, 6.f});
}

TEST_CASE("mean and max after transpose reduce logical elements") {
    const tensor::Tensor<float> t({2, 3}, std::vector<float>{1.f, 2.f, 3.f, 4.f, 5.f, 6.f}, false);
    CHECK(t.transpose().mean().data()[0] == doctest::Approx(3.5f));
    CHECK(t.transpose().max().data()[0] == doctest::Approx(6.f));
    CHECK(t.transpose().min().data()[0] == doctest::Approx(1.f));
}

TEST_CASE("sum(dim) after transpose reduces the logical axes") {
    const tensor::Tensor<float> t({2, 3}, std::vector<float>{1.f, 2.f, 3.f, 4.f, 5.f, 6.f}, false);
    // transpose is [3, 2]; sum dim 1 → [1+4, 2+5, 3+6]
    const auto s = t.transpose().sum(1);
    CHECK(s.shape() == std::vector<int64_t>{3});
    CHECK(s.data() == std::vector<float>{5.f, 7.f, 9.f});
}

TEST_CASE("mean/max/min/softmax after transpose reduce the logical axes") {
    const tensor::Tensor<float> t({2, 3}, std::vector<float>{1.f, 2.f, 3.f, 4.f, 5.f, 6.f}, false);
    const auto tr = t.transpose();

    const auto m = tr.mean(1);
    CHECK(m.shape() == std::vector<int64_t>{3});
    CHECK(m.data()[0] == doctest::Approx(2.5f));
    CHECK(m.data()[1] == doctest::Approx(3.5f));
    CHECK(m.data()[2] == doctest::Approx(4.5f));

    const auto mx = tr.max(1);
    CHECK(mx.data() == std::vector<float>{4.f, 5.f, 6.f});

    const auto mn = tr.min(1);
    CHECK(mn.data() == std::vector<float>{1.f, 2.f, 3.f});

    const auto s = tr.softmax(1);
    CHECK(s.shape() == (std::vector<int64_t>{3, 2}));
    const float den0 = std::exp(1.f) + std::exp(4.f);
    CHECK(s.data()[0] == doctest::Approx(std::exp(1.f) / den0));
    CHECK(s.data()[1] == doctest::Approx(std::exp(4.f) / den0));
}

TEST_CASE("add after broadcast_to uses the repeated logical values") {
    const tensor::Tensor<float> col({3, 1}, std::vector<float>{1.f, 2.f, 3.f}, false);
    const tensor::Tensor<float> z({3, 2}, std::vector<float>(6, 0.f), false);
    const auto r = col.broadcast_to({3, 2}).add(z);
    CHECK(r.data() == std::vector<float>{1.f, 1.f, 2.f, 2.f, 3.f, 3.f});
}

TEST_CASE("relu after narrow uses only the slice") {
    const tensor::Tensor<float> t({2, 3}, std::vector<float>{1.f, -2.f, 3.f, -4.f, 5.f, -6.f}, false);
    // columns 1..2: [-2, 3] / [5, -6] → relu packed {0, 3, 5, 0}
    const auto r = t.narrow(1, 1, 2).relu();
    CHECK(r.shape() == std::vector<int64_t>{2, 2});
    CHECK(r.data() == std::vector<float>{0.f, 3.f, 5.f, 0.f});
}

TEST_CASE("l1_loss after transpose compares logical order") {
    const tensor::Tensor<float> t({2, 3}, std::vector<float>{1.f, 2.f, 3.f, 4.f, 5.f, 6.f}, false);
    const tensor::Tensor<float> tgt({3, 2}, std::vector<float>{1.f, 4.f, 2.f, 5.f, 3.f, 6.f}, false);
    CHECK(ops::l1_loss(t.transpose(), tgt).data()[0] == doctest::Approx(0.f));
}

TEST_CASE("add after transpose backprops onto the leaf in storage order") {
    tensor::Tensor<float> x({2, 3}, std::vector<float>{1.f, 2.f, 3.f, 4.f, 5.f, 6.f}, true);
    const tensor::Tensor<float> ones({3, 2}, std::vector<float>(6, 1.f), false);
    x.transpose().add(ones).sum().backward();
    REQUIRE(x.grad() != nullptr);
    for (int64_t i = 0; i < 6; ++i)
        CHECK(x.grad()->data()[static_cast<size_t>(i)] == doctest::Approx(1.f));
}

TEST_CASE("multiply after transpose backprops the other operand in logical order") {
    tensor::Tensor<float> x({2, 3}, std::vector<float>{1.f, 2.f, 3.f, 4.f, 5.f, 6.f}, true);
    const tensor::Tensor<float> scale({3, 2}, std::vector<float>{10.f, 20.f, 30.f, 40.f, 50.f, 60.f}, false);
    x.transpose().multiply(scale).sum().backward();
    REQUIRE(x.grad() != nullptr);
    // dL/dx in owner layout: scale written back through the transpose
    // scale [3,2] packed {10,20,30,40,50,60} → owner [2,3] {10,30,50, 20,40,60}
    const std::vector<float> expected{10.f, 30.f, 50.f, 20.f, 40.f, 60.f};
    for (size_t i = 0; i < expected.size(); ++i)
        CHECK(x.grad()->data()[i] == doctest::Approx(expected[i]));
}
