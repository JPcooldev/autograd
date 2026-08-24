#include <cstdint>
#include <stdexcept>
#include <vector>

#include "../doctest/doctest.h"

#include "../../src/autograd/autograd.h"

TEST_CASE("Tensor transpose swaps shape and strides for rank-2 tensor") {
    const tensor::Tensor<float> matrix({2, 3}, std::vector<float>{1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F}, false);

    const tensor::Tensor<float> transposed = matrix.transpose();

    CHECK(transposed.shape() == std::vector<int64_t>{3, 2});
    CHECK(transposed.strides() == std::vector<int64_t>{1, 3});
    CHECK(transposed.data() == matrix.data());
    CHECK(transposed.is_view());
}

TEST_CASE("Tensor transpose supports explicit dimensions for rank-3 tensor") {
    const tensor::Tensor<int32_t> tensor3d({2, 3, 4}, std::vector<int32_t>(24, 1), false);

    const tensor::Tensor<int32_t> transposed = tensor3d.transpose(0, 2);

    CHECK(transposed.shape() == std::vector<int64_t>{4, 3, 2});
    CHECK(transposed.strides() == std::vector<int64_t>{1, 4, 12});
}

TEST_CASE("Tensor transpose supports negative dimensions") {
    const tensor::Tensor<double> tensor3d({2, 3, 4}, std::vector<double>(24, 0.5), false);

    const tensor::Tensor<double> transposed = tensor3d.transpose(-1, -3);

    CHECK(transposed.shape() == std::vector<int64_t>{4, 3, 2});
    CHECK(transposed.strides() == std::vector<int64_t>{1, 4, 12});
}

TEST_CASE("Tensor transpose validates ranks and dimensions") {
    const tensor::Tensor<float> vector({4}, std::vector<float>{1.0F, 2.0F, 3.0F, 4.0F}, false);

    CHECK_THROWS_AS(vector.transpose(), std::invalid_argument);
    CHECK_THROWS_AS(vector.transpose(0, 1), std::out_of_range);

    const tensor::Tensor<float> scalar({}, false);
    CHECK_THROWS_AS(scalar.transpose(0, 0), std::invalid_argument);
}

TEST_CASE("Tensor transpose creates dedicated backward node with inverse transpose") {
    const tensor::Tensor<float> input({2, 3}, std::vector<float>{1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F}, true);
    const tensor::Tensor<float> transposed = input.transpose(0, 1);

    REQUIRE(transposed.requires_grad());
    REQUIRE(transposed.grad_fn().get() != nullptr);
    REQUIRE(transposed.grad_fn()->next_edges.size() == 1);
    // input is a leaf: with AccumulateGrad eliminated, the edge is nullptr and
    // the engine accumulates directly via saved_tensors[0].
    CHECK(transposed.grad_fn()->next_edges[0].get() == nullptr);

    const tensor::Tensor<float> propagated_grad = tensor::Tensor<float>::ones(transposed.shape(), false);
    const std::vector<tensor::Tensor<float>> grads = transposed.grad_fn()->apply(propagated_grad);

    REQUIRE(grads.size() == 1);
    CHECK(grads[0].shape() == input.shape());
    CHECK(grads[0].strides() == std::vector<int64_t>{1, 2});
    CHECK(grads[0].is_view());
}

TEST_CASE("reshape reinterprets contiguous storage") {
    const tensor::Tensor<float> t({2, 3}, std::vector<float>{1.f, 2.f, 3.f, 4.f, 5.f, 6.f}, false);
    const auto r = t.reshape({3, 2});

    CHECK(r.shape() == std::vector<int64_t>{3, 2});
    CHECK(r.is_contiguous());
    CHECK(r.is_view());
    CHECK(r.data().data() == t.data().data());
    CHECK(static_cast<float>(r[1][0]) == doctest::Approx(3.f));
}

TEST_CASE("view matches reshape on a contiguous tensor") {
    const tensor::Tensor<float> t({2, 3}, std::vector<float>{1.f, 2.f, 3.f, 4.f, 5.f, 6.f}, false);
    const auto v = t.view({6});

    CHECK(v.shape() == std::vector<int64_t>{6});
    CHECK(v.numel() == 6);
    CHECK(v.data().data() == t.data().data());
}

TEST_CASE("reshape and flatten reject non-contiguous tensors") {
    const tensor::Tensor<float> t({2, 3}, std::vector<float>{1.f, 2.f, 3.f, 4.f, 5.f, 6.f}, false);
    const auto tr = t.transpose();

    CHECK_THROWS_AS(tr.reshape({6}), std::invalid_argument);
    CHECK_THROWS_AS(tr.flatten(), std::invalid_argument);
}

TEST_CASE("flatten merges a contiguous range of axes") {
    const tensor::Tensor<float> t({2, 3, 4}, std::vector<float>(24, 1.f), false);
    const auto f = t.flatten(1, -1);

    CHECK(f.shape() == std::vector<int64_t>{2, 12});
    CHECK(f.is_contiguous());
    CHECK(f.data().data() == t.data().data());
}

TEST_CASE("squeeze drops size-1 axes; squeeze(dim) is a no-op when size != 1") {
    const tensor::Tensor<float> t({1, 3, 1}, std::vector<float>{1.f, 2.f, 3.f}, false);

    CHECK(t.squeeze().shape() == std::vector<int64_t>{3});
    CHECK(t.squeeze(0).shape() == std::vector<int64_t>{3, 1});
    CHECK(t.squeeze(1).shape() == std::vector<int64_t>{1, 3, 1});
}

TEST_CASE("unsqueeze inserts a size-1 axis") {
    const tensor::Tensor<float> t({3}, std::vector<float>{1.f, 2.f, 3.f}, false);
    const auto u = t.unsqueeze(0);

    CHECK(u.shape() == std::vector<int64_t>{1, 3});
    CHECK(static_cast<float>(u[0][1]) == doctest::Approx(2.f));
}

TEST_CASE("broadcast_to expands with stride 0") {
    const tensor::Tensor<float> t({1, 3}, std::vector<float>{1.f, 2.f, 3.f}, false);
    const auto b = t.broadcast_to({2, 3});

    CHECK(b.shape() == std::vector<int64_t>{2, 3});
    CHECK(b.strides()[0] == 0);
    CHECK(b.is_view());
    CHECK(static_cast<float>(b[0][2]) == doctest::Approx(3.f));
    CHECK(static_cast<float>(b[1][2]) == doctest::Approx(3.f));
}

TEST_CASE("contiguous is a no-copy fast path on a packed tensor") {
    const tensor::Tensor<float> t({2, 3}, std::vector<float>{1.f, 2.f, 3.f, 4.f, 5.f, 6.f}, true);
    const auto c = t.contiguous();

    CHECK(c.is_contiguous());
    CHECK(c.offset() == 0);
    CHECK(c.data().data() == t.data().data());
    CHECK(c.grad_fn().get() == t.grad_fn().get());
}

TEST_CASE("contiguous of a transpose copies logical order into a new buffer") {
    const tensor::Tensor<float> t({2, 3}, std::vector<float>{1.f, 2.f, 3.f, 4.f, 5.f, 6.f}, false);
    const auto c = t.transpose().contiguous();

    CHECK(c.shape() == std::vector<int64_t>{3, 2});
    CHECK(c.is_contiguous());
    CHECK(c.offset() == 0);
    CHECK(c.data() != t.data());
    CHECK(c.data() == std::vector<float>{1.f, 4.f, 2.f, 5.f, 3.f, 6.f});
}

TEST_CASE("contiguous after transpose then reshape succeeds") {
    const tensor::Tensor<float> t({2, 3}, std::vector<float>{1.f, 2.f, 3.f, 4.f, 5.f, 6.f}, false);
    const auto r = t.transpose().contiguous().reshape({6});
    CHECK(r.shape() == std::vector<int64_t>{6});
    CHECK(r.data() == std::vector<float>{1.f, 4.f, 2.f, 5.f, 3.f, 6.f});
}

TEST_CASE("sum of transpose().contiguous() backprops all-ones into the leaf") {
    tensor::Tensor<float> x({2, 3}, std::vector<float>{1.f, 2.f, 3.f, 4.f, 5.f, 6.f}, true);
    x.transpose().contiguous().sum().backward();

    REQUIRE(x.grad() != nullptr);
    REQUIRE(x.grad()->numel() == 6);
    for (int64_t i = 0; i < 6; ++i)
        CHECK(x.grad()->data()[static_cast<size_t>(i)] == doctest::Approx(1.f));
}

TEST_CASE("narrow is a view along dim") {
    const tensor::Tensor<float> t({2, 3}, std::vector<float>{1.f, 2.f, 3.f, 4.f, 5.f, 6.f}, false);
    const auto n = ops::narrow(t, 1, 1, 2);
    CHECK(n.shape() == std::vector<int64_t>{2, 2});
    CHECK(n.is_view());
    CHECK(static_cast<float>(n[0][0]) == doctest::Approx(2.f));
    CHECK(static_cast<float>(n[1][1]) == doctest::Approx(6.f));
}

TEST_CASE("narrow backward scatters into zeros") {
    tensor::Tensor<float> x({2, 3}, std::vector<float>{1.f, 2.f, 3.f, 4.f, 5.f, 6.f}, true);
    x.narrow(1, 1, 2).sum().backward();
    REQUIRE(x.grad() != nullptr);
    const std::vector<float> expected{0.f, 1.f, 1.f, 0.f, 1.f, 1.f};
    for (size_t i = 0; i < expected.size(); ++i)
        CHECK(x.grad()->data()[i] == doctest::Approx(expected[i]));
}

TEST_CASE("cat concatenates along dim") {
    const tensor::Tensor<float> a({2, 2}, std::vector<float>{1.f, 2.f, 3.f, 4.f}, false);
    const tensor::Tensor<float> b({2, 1}, std::vector<float>{5.f, 6.f}, false);
    const auto c = ops::cat({a, b}, 1);
    CHECK(c.shape() == std::vector<int64_t>{2, 3});
    CHECK(c.data() == std::vector<float>{1.f, 2.f, 5.f, 3.f, 4.f, 6.f});
}

TEST_CASE("cat backward splits the gradient") {
    tensor::Tensor<float> a({2}, std::vector<float>{1.f, 2.f}, true);
    tensor::Tensor<float> b({1}, std::vector<float>{3.f}, true);
    ops::cat({a, b}, 0).sum().backward();
    REQUIRE(a.grad() != nullptr);
    REQUIRE(b.grad() != nullptr);
    CHECK(a.grad()->data()[0] == doctest::Approx(1.f));
    CHECK(a.grad()->data()[1] == doctest::Approx(1.f));
    CHECK(b.grad()->data()[0] == doctest::Approx(1.f));
}

