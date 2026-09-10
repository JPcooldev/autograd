/*
 * to_vector, copy_, copy construction, and Tensor::alias.
 *
 * - to_vector on a packed tensor and on a transpose
 * - copy_ keeps dest storage and requires_grad
 * - copy_ from a view writes logical order
 * - shape mismatch throws
 * - scalar copy_
 * - copy_ keeps leaf identity so backward still writes dest.grad
 * - copy constructor shares storage and GradStorage
 * - alias of a leaf shares GradStorage and clears grad_fn
 */

#include <stdexcept>
#include <vector>

#include "../doctest/doctest.h"

#include "../../src/autograd/autograd.h"

TEST_CASE("to_vector packs a contiguous tensor in row-major order") {
    tensor::Tensor<float> x({2, 3}, {1.f, 2.f, 3.f, 4.f, 5.f, 6.f}, false);
    CHECK(x.to_vector() == std::vector<float>{1.f, 2.f, 3.f, 4.f, 5.f, 6.f});
}

TEST_CASE("to_vector gathers a transpose view in logical order") {
    tensor::Tensor<float> x({2, 3}, {1.f, 2.f, 3.f, 4.f, 5.f, 6.f}, false);
    auto t = x.transpose();
    CHECK(t.shape() == std::vector<int64_t>{3, 2});
    CHECK(t.to_vector() == std::vector<float>{1.f, 4.f, 2.f, 5.f, 3.f, 6.f});
}

TEST_CASE("copy_ writes values without replacing storage") {
    tensor::Tensor<float> dest({2, 2}, 0.f, true);
    float* storage = dest.data().data();
    tensor::Tensor<float> src({2, 2}, {1.f, 2.f, 3.f, 4.f}, false);
    dest.copy_(src);
    CHECK(dest.data().data() == storage);
    CHECK(dest.requires_grad() == true);
    CHECK(dest.to_vector() == std::vector<float>{1.f, 2.f, 3.f, 4.f});
}

TEST_CASE("copy_ from a view writes packed logical values") {
    tensor::Tensor<float> src({2, 3}, {1.f, 2.f, 3.f, 4.f, 5.f, 6.f}, false);
    tensor::Tensor<float> dest({3, 2}, 0.f, false);
    dest.copy_(src.transpose());
    CHECK(dest.to_vector() == std::vector<float>{1.f, 4.f, 2.f, 5.f, 3.f, 6.f});
}

TEST_CASE("copy_ throws on shape mismatch") {
    tensor::Tensor<float> dest({2}, 0.f, false);
    tensor::Tensor<float> src({3}, 1.f, false);
    CHECK_THROWS_AS(dest.copy_(src), std::invalid_argument);
}

TEST_CASE("copy_ of a scalar tensor") {
    tensor::Tensor<float> dest({}, std::vector<float>{0.f}, false);
    tensor::Tensor<float> src({}, std::vector<float>{3.5f}, false);
    dest.copy_(src);
    CHECK(dest.to_vector()[0] == doctest::Approx(3.5f));
}

TEST_CASE("copy_ keeps leaf identity so backward still writes dest.grad") {
    tensor::Tensor<float> dest({2}, {0.f, 0.f}, true);
    dest.copy_(tensor::Tensor<float>({2}, {1.f, 2.f}, false));
    CHECK(dest.is_leaf());
    CHECK(dest.grad_fn().get() == nullptr);
    dest.sum().backward();
    REQUIRE(dest.grad() != nullptr);
    CHECK(dest.grad()->data() == std::vector<float>{1.f, 1.f});
}

TEST_CASE("copy constructor shares storage and leaf GradStorage") {
    tensor::Tensor<float> a({2}, {1.f, 2.f}, true);
    tensor::Tensor<float> b = a;
    CHECK(b.data().data() == a.data().data());
    b.data()[0] = 9.f;
    CHECK(a.data()[0] == doctest::Approx(9.f));
    b.sum().backward();
    REQUIRE(a.grad() != nullptr);
    CHECK(a.grad()->data() == std::vector<float>{1.f, 1.f});
    CHECK(b.grad() == a.grad());
}

TEST_CASE("alias of a leaf shares GradStorage and clears grad_fn") {
    tensor::Tensor<float> a({2}, {1.f, 2.f}, true);
    auto b = tensor::Tensor<float>::alias(a);
    CHECK(b.is_leaf());
    CHECK(b.grad_fn().get() == nullptr);
    CHECK(b.data().data() == a.data().data());
    b.sum().backward();
    REQUIRE(a.grad() != nullptr);
    CHECK(a.grad()->data() == std::vector<float>{1.f, 1.f});
}
