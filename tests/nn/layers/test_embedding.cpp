/*
 * nn::Embedding: table size, row lookup, scatter-add grad, ctor checks.
 *
 * - weight is {V, D} with requires_grad
 * - forward looks up rows; backward fills weight.grad (repeats accumulate)
 * - non-positive V or D throws
 */

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "../../doctest/doctest.h"

#include "../../../src/autograd/autograd.h"
#include "../../../src/nn/layers/embedding.h"

TEST_CASE("Embedding table is {V, D} with requires_grad") {
    nn::Embedding<float> emb(5, 3);
    CHECK(emb.num_embeddings() == 5);
    CHECK(emb.embedding_dim() == 3);
    CHECK(emb.weight.shape() == std::vector<int64_t>{5, 3});
    CHECK(emb.weight.requires_grad());
    CHECK(emb.parameters().size() == 1);
}

TEST_CASE("Embedding forward looks up rows and backward fills weight.grad") {
    nn::Embedding<float> emb(4, 2);
    emb.weight.data() = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f};
    tensor::Tensor<int32_t> idx({2, 2}, std::vector<int32_t>{0, 3, 3, 1}, false);
    auto y = emb.forward(idx);
    CHECK(y.shape() == std::vector<int64_t>{2, 2, 2});
    CHECK(y.data()[0] == doctest::Approx(1.f));
    CHECK(y.data()[1] == doctest::Approx(2.f));
    CHECK(y.data()[2] == doctest::Approx(7.f));
    CHECK(y.data()[3] == doctest::Approx(8.f));
    CHECK(y.data()[4] == doctest::Approx(7.f));
    CHECK(y.data()[5] == doctest::Approx(8.f));
    CHECK(y.data()[6] == doctest::Approx(3.f));
    CHECK(y.data()[7] == doctest::Approx(4.f));
    y.sum().backward();
    REQUIRE(emb.weight.grad() != nullptr);
    CHECK(emb.weight.grad()->data()[0] == doctest::Approx(1.f));
    CHECK(emb.weight.grad()->data()[6] == doctest::Approx(2.f));
}

TEST_CASE("Embedding rejects non-positive sizes") {
    CHECK_THROWS_AS(nn::Embedding<float>(0, 4), std::invalid_argument);
    CHECK_THROWS_AS(nn::Embedding<float>(4, 0), std::invalid_argument);
}
