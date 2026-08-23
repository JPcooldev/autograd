#include <cstdint>
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

