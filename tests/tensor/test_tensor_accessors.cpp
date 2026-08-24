#include <cstdint>
#include <vector>

#include "../doctest/doctest.h"

#include "../../src/autograd/autograd.h"

TEST_CASE("accessors report rank, ndim, numel, and shape") {
    const tensor::Tensor<float> t({2, 3}, std::vector<float>{1.f, 2.f, 3.f, 4.f, 5.f, 6.f}, false);

    CHECK(t.rank() == 2);
    CHECK(t.ndim() == 2);
    CHECK(t.numel() == 6);
    CHECK(t.shape() == std::vector<int64_t>{2, 3});
}

TEST_CASE("contiguous owner has unit innermost stride and zero offset") {
    const tensor::Tensor<float> t({2, 3}, std::vector<float>{1.f, 2.f, 3.f, 4.f, 5.f, 6.f}, false);

    CHECK(t.strides() == std::vector<int64_t>{3, 1});
    CHECK(t.offset() == 0);
    CHECK(t.is_contiguous());
    CHECK_FALSE(t.is_view());
}

TEST_CASE("dtype and requires_grad accessors") {
    const tensor::Tensor<float> f({2}, std::vector<float>{1.f, 2.f}, true);
    const tensor::Tensor<int32_t> i({2}, std::vector<int32_t>{1, 2}, true);

    CHECK(f.dtype() == tensor::Dtype::Float32);
    CHECK(f.requires_grad());
    CHECK(i.dtype() == tensor::Dtype::Int32);
    CHECK_FALSE(i.requires_grad());
}

TEST_CASE("leaf has null grad_fn; data() is the backing buffer") {
    const tensor::Tensor<float> t({2}, std::vector<float>{1.f, 2.f}, true);

    CHECK(t.grad_fn().get() == nullptr);
    CHECK(t.is_leaf());
    CHECK(t.data() == std::vector<float>{1.f, 2.f});
}

TEST_CASE("transpose view shares storage and is not contiguous") {
    const tensor::Tensor<float> t({2, 3}, std::vector<float>{1.f, 2.f, 3.f, 4.f, 5.f, 6.f}, false);
    const auto view = t.transpose();

    CHECK(view.is_view());
    CHECK_FALSE(view.is_contiguous());
    CHECK(view.shape() == std::vector<int64_t>{3, 2});
    CHECK(view.strides() == std::vector<int64_t>{1, 3});
    CHECK(view.offset() == 0);
    CHECK(view.data().data() == t.data().data());
    CHECK(static_cast<float>(view[0][1]) == doctest::Approx(4.f));
}
