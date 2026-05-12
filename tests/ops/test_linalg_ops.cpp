#include <stdexcept>
#include <vector>

#include "../doctest/doctest.h"

#include "../../src/autograd/autograd.h"

// ─── dot: forward ─────────────────────────────────────────────────────────────

TEST_CASE("dot produces scalar tensor of shape {}") {
    const tensor::Tensor<float> x({3}, std::vector<float>{1.f, 2.f, 3.f}, false);
    const tensor::Tensor<float> y({3}, std::vector<float>{4.f, 5.f, 6.f}, false);
    const auto result = ops::dot(x, y);

    REQUIRE(result.shape() == std::vector<int64_t>{});
    REQUIRE(result.numel() == 1);
}

TEST_CASE("dot computes correct value") {
    // [1,2,3] · [4,5,6] = 4 + 10 + 18 = 32
    const tensor::Tensor<float> x({3}, std::vector<float>{1.f, 2.f, 3.f}, false);
    const tensor::Tensor<float> y({3}, std::vector<float>{4.f, 5.f, 6.f}, false);
    const auto result = ops::dot(x, y);

    CHECK(result.data()[0] == doctest::Approx(32.f));
}

TEST_CASE("dot method gives same result as free function") {
    const tensor::Tensor<float> x({4}, std::vector<float>{1.f, -1.f, 2.f, -2.f}, false);
    const tensor::Tensor<float> y({4}, std::vector<float>{3.f, 3.f, 4.f, 4.f}, false);

    CHECK(x.dot(y).data()[0] == doctest::Approx(ops::dot(x, y).data()[0]));
}

TEST_CASE("dot works with double") {
    const tensor::Tensor<double> x({2}, std::vector<double>{1.5, 2.5}, false);
    const tensor::Tensor<double> y({2}, std::vector<double>{2.0, 4.0}, false);
    // 1.5*2.0 + 2.5*4.0 = 3.0 + 10.0 = 13.0
    CHECK(x.dot(y).data()[0] == doctest::Approx(13.0));
}

TEST_CASE("dot throws on non-1D input") {
    const tensor::Tensor<float> a({2, 2}, std::vector<float>{1.f, 2.f, 3.f, 4.f}, false);
    const tensor::Tensor<float> b({4},    std::vector<float>{1.f, 2.f, 3.f, 4.f}, false);

    CHECK_THROWS_AS(ops::dot(a, b), std::invalid_argument);
    CHECK_THROWS_AS(ops::dot(b, a), std::invalid_argument);
}

TEST_CASE("dot throws on size mismatch") {
    const tensor::Tensor<float> x({3}, std::vector<float>{1.f, 2.f, 3.f}, false);
    const tensor::Tensor<float> y({4}, std::vector<float>{1.f, 2.f, 3.f, 4.f}, false);

    CHECK_THROWS_AS(ops::dot(x, y), std::invalid_argument);
}

// ─── dot: autograd ────────────────────────────────────────────────────────────

TEST_CASE("dot produces no grad_fn when both inputs are no-grad") {
    const tensor::Tensor<float> x({3}, std::vector<float>{1.f, 2.f, 3.f}, false);
    const tensor::Tensor<float> y({3}, std::vector<float>{4.f, 5.f, 6.f}, false);

    CHECK(ops::dot(x, y).grad_fn().get() == nullptr);
}

TEST_CASE("dot produces grad_fn when either input requires grad") {
    const tensor::Tensor<float> x({3}, std::vector<float>{1.f, 2.f, 3.f}, true);
    const tensor::Tensor<float> y({3}, std::vector<float>{4.f, 5.f, 6.f}, false);

    CHECK(ops::dot(x, y).grad_fn().get() != nullptr);
}

TEST_CASE("DotBackward propagates correct gradients") {
    // x = [1, 2, 3], y = [4, 5, 6], grad_scalar = 2
    // grad_x[i] = 2 * y[i] -> [8, 10, 12]
    // grad_y[i] = 2 * x[i] -> [2, 4, 6]
    const tensor::Tensor<float> x({3}, std::vector<float>{1.f, 2.f, 3.f}, true);
    const tensor::Tensor<float> y({3}, std::vector<float>{4.f, 5.f, 6.f}, true);
    const auto result = ops::dot(x, y);

    REQUIRE(result.grad_fn().get() != nullptr);

    const tensor::Tensor<float> propagated({}, std::vector<float>{2.f}, false);
    const auto grads = result.grad_fn()->apply(propagated);

    REQUIRE(grads.size() == 2);
    REQUIRE(grads[0].shape() == std::vector<int64_t>{3});
    REQUIRE(grads[1].shape() == std::vector<int64_t>{3});

    CHECK(grads[0].data()[0] == doctest::Approx(8.f));
    CHECK(grads[0].data()[1] == doctest::Approx(10.f));
    CHECK(grads[0].data()[2] == doctest::Approx(12.f));

    CHECK(grads[1].data()[0] == doctest::Approx(2.f));
    CHECK(grads[1].data()[1] == doctest::Approx(4.f));
    CHECK(grads[1].data()[2] == doctest::Approx(6.f));
}

TEST_CASE("dot wires next_edges to input grad_fn") {
    const tensor::Tensor<float> a({3}, std::vector<float>{1.f, 1.f, 1.f}, true);
    const tensor::Tensor<float> b({3}, std::vector<float>{2.f, 2.f, 2.f}, true);
    const auto mid = a.add(b);         // has AddBackward grad_fn
    const tensor::Tensor<float> c({3}, std::vector<float>{1.f, 0.f, 0.f}, false);
    const auto result = ops::dot(mid, c);

    REQUIRE(result.grad_fn().get() != nullptr);
    REQUIRE(result.grad_fn()->next_edges.size() == 2);
    CHECK(result.grad_fn()->next_edges[0].get() == mid.grad_fn().get());
}

// ─── matmul: forward ──────────────────────────────────────────────────────────

TEST_CASE("matmul produces correct output shape") {
    // (2,3) @ (3,4) -> (2,4)
    const tensor::Tensor<float> A({2, 3}, std::vector<float>{1.f,2.f,3.f, 4.f,5.f,6.f}, false);
    const tensor::Tensor<float> B({3, 4}, std::vector<float>{
        1.f,0.f,0.f,0.f,
        0.f,1.f,0.f,0.f,
        0.f,0.f,1.f,0.f}, false);
    const auto C = ops::matmul(A, B);

    REQUIRE(C.shape() == std::vector<int64_t>{2, 4});
}

TEST_CASE("matmul computes correct values: (2,3) @ (3,2)") {
    // A = [[1,2,3],[4,5,6]], B = [[7,8],[9,10],[11,12]]
    // C[0,0] = 7+18+33 = 58,  C[0,1] = 8+20+36 = 64
    // C[1,0] = 28+45+66 = 139, C[1,1] = 32+50+72 = 154
    const tensor::Tensor<float> A({2, 3},
        std::vector<float>{1.f,2.f,3.f, 4.f,5.f,6.f}, false);
    const tensor::Tensor<float> B({3, 2},
        std::vector<float>{7.f,8.f, 9.f,10.f, 11.f,12.f}, false);
    const auto C = ops::matmul(A, B);

    REQUIRE(C.shape() == std::vector<int64_t>{2, 2});
    CHECK(C.data()[0] == doctest::Approx(58.f));
    CHECK(C.data()[1] == doctest::Approx(64.f));
    CHECK(C.data()[2] == doctest::Approx(139.f));
    CHECK(C.data()[3] == doctest::Approx(154.f));
}

TEST_CASE("matmul with identity matrix leaves the other operand unchanged") {
    const tensor::Tensor<float> A({2, 2},
        std::vector<float>{3.f, 7.f, -1.f, 5.f}, false);
    const tensor::Tensor<float> I({2, 2},
        std::vector<float>{1.f, 0.f, 0.f, 1.f}, false);
    const auto C = ops::matmul(A, I);

    for (size_t i = 0; i < 4; ++i)
        CHECK(C.data()[i] == doctest::Approx(A.data()[i]));
}

TEST_CASE("matmul method gives same result as free function") {
    const tensor::Tensor<float> A({2, 2}, std::vector<float>{1.f,2.f,3.f,4.f}, false);
    const tensor::Tensor<float> B({2, 2}, std::vector<float>{5.f,6.f,7.f,8.f}, false);
    const auto via_method = A.matmul(B);
    const auto via_free   = ops::matmul(A, B);

    for (size_t i = 0; i < 4; ++i)
        CHECK(via_method.data()[i] == doctest::Approx(via_free.data()[i]));
}

TEST_CASE("matmul works with double") {
    const tensor::Tensor<double> A({2, 2},
        std::vector<double>{1.0, 2.0, 3.0, 4.0}, false);
    const tensor::Tensor<double> B({2, 2},
        std::vector<double>{1.0, 0.0, 0.0, 1.0}, false);
    const auto C = A.matmul(B);

    for (size_t i = 0; i < 4; ++i)
        CHECK(C.data()[i] == doctest::Approx(A.data()[i]));
}

TEST_CASE("matmul throws when first argument is not 2D") {
    const tensor::Tensor<float> A({6},    std::vector<float>(6, 1.f), false);
    const tensor::Tensor<float> B({2, 3}, std::vector<float>(6, 1.f), false);

    CHECK_THROWS_AS(ops::matmul(A, B), std::invalid_argument);
}

TEST_CASE("matmul throws when second argument is not 2D") {
    const tensor::Tensor<float> A({2, 3}, std::vector<float>(6, 1.f), false);
    const tensor::Tensor<float> B({6},    std::vector<float>(6, 1.f), false);

    CHECK_THROWS_AS(ops::matmul(A, B), std::invalid_argument);
}

TEST_CASE("matmul throws on inner dimension mismatch") {
    const tensor::Tensor<float> A({2, 3}, std::vector<float>(6, 1.f), false);
    const tensor::Tensor<float> B({4, 2}, std::vector<float>(8, 1.f), false);

    CHECK_THROWS_AS(ops::matmul(A, B), std::invalid_argument);
}

// ─── matmul: stride-aware (transposed input) ──────────────────────────────────

TEST_CASE("matmul with transposed input uses stride-aware access") {
    // A = [[1,2],[3,4]], A^T = [[1,3],[2,4]] (non-contiguous view)
    // B = [[5,6],[7,8]]
    // A^T @ B:
    //   C[0,0] = 1*5 + 3*7 = 26,  C[0,1] = 1*6 + 3*8 = 30
    //   C[1,0] = 2*5 + 4*7 = 38,  C[1,1] = 2*6 + 4*8 = 44
    const tensor::Tensor<float> A({2, 2},
        std::vector<float>{1.f, 2.f, 3.f, 4.f}, false);
    const tensor::Tensor<float> B({2, 2},
        std::vector<float>{5.f, 6.f, 7.f, 8.f}, false);

    const auto A_t = A.transpose();  // non-contiguous: strides = [1, 2]
    const auto C   = ops::matmul(A_t, B);

    REQUIRE(C.shape() == std::vector<int64_t>{2, 2});
    CHECK(C.data()[0] == doctest::Approx(26.f));
    CHECK(C.data()[1] == doctest::Approx(30.f));
    CHECK(C.data()[2] == doctest::Approx(38.f));
    CHECK(C.data()[3] == doctest::Approx(44.f));
}

// ─── matmul: autograd ─────────────────────────────────────────────────────────

TEST_CASE("matmul produces no grad_fn when all inputs are no-grad") {
    const tensor::Tensor<float> A({2, 2}, std::vector<float>{1.f,2.f,3.f,4.f}, false);
    const tensor::Tensor<float> B({2, 2}, std::vector<float>{5.f,6.f,7.f,8.f}, false);

    CHECK(ops::matmul(A, B).grad_fn().get() == nullptr);
}

TEST_CASE("matmul produces grad_fn when either input requires grad") {
    const tensor::Tensor<float> A({2, 2}, std::vector<float>{1.f,2.f,3.f,4.f}, true);
    const tensor::Tensor<float> B({2, 2}, std::vector<float>{5.f,6.f,7.f,8.f}, false);

    CHECK(ops::matmul(A, B).grad_fn().get() != nullptr);
}

TEST_CASE("MatmulBackward propagates correct gradients") {
    // A = [[1,2],[3,4]], B = [[5,6],[7,8]], grad = identity [[1,0],[0,1]]
    // dA = grad @ B^T = I @ B^T = B^T = [[5,7],[6,8]]
    // dB = A^T @ grad = A^T @ I = A^T = [[1,3],[2,4]]
    const tensor::Tensor<float> A({2, 2},
        std::vector<float>{1.f, 2.f, 3.f, 4.f}, true);
    const tensor::Tensor<float> B({2, 2},
        std::vector<float>{5.f, 6.f, 7.f, 8.f}, true);
    const auto C = ops::matmul(A, B);

    REQUIRE(C.grad_fn().get() != nullptr);

    const tensor::Tensor<float> grad({2, 2},
        std::vector<float>{1.f, 0.f, 0.f, 1.f}, false);
    const auto grads = C.grad_fn()->apply(grad);

    REQUIRE(grads.size() == 2);
    REQUIRE(grads[0].shape() == std::vector<int64_t>{2, 2});  // dA
    REQUIRE(grads[1].shape() == std::vector<int64_t>{2, 2});  // dB

    // dA = B^T = [[5,7],[6,8]]
    CHECK(grads[0].data()[0] == doctest::Approx(5.f));
    CHECK(grads[0].data()[1] == doctest::Approx(7.f));
    CHECK(grads[0].data()[2] == doctest::Approx(6.f));
    CHECK(grads[0].data()[3] == doctest::Approx(8.f));

    // dB = A^T = [[1,3],[2,4]]
    CHECK(grads[1].data()[0] == doctest::Approx(1.f));
    CHECK(grads[1].data()[1] == doctest::Approx(3.f));
    CHECK(grads[1].data()[2] == doctest::Approx(2.f));
    CHECK(grads[1].data()[3] == doctest::Approx(4.f));
}

TEST_CASE("MatmulBackward saves contiguous copies of transposed inputs") {
    // A_t = [[1,3],[2,4]] (transpose of [[1,2],[3,4]], non-contiguous)
    // B   = [[5,6],[7,8]]
    // C = A_t @ B, grad = [[1,0],[0,1]]
    // dA_t = grad @ B^T = B^T = [[5,7],[6,8]]
    // dB   = A_t^T @ grad = [[1,2],[3,4]]
    const tensor::Tensor<float> A_raw({2, 2},
        std::vector<float>{1.f, 2.f, 3.f, 4.f}, true);
    const tensor::Tensor<float> B({2, 2},
        std::vector<float>{5.f, 6.f, 7.f, 8.f}, true);

    const auto A_t = A_raw.transpose();
    const auto C   = ops::matmul(A_t, B);

    REQUIRE(C.grad_fn().get() != nullptr);

    const tensor::Tensor<float> grad({2, 2},
        std::vector<float>{1.f, 0.f, 0.f, 1.f}, false);
    const auto grads = C.grad_fn()->apply(grad);

    REQUIRE(grads.size() == 2);

    // dA_t shape == A_t's shape == [2,2]
    REQUIRE(grads[0].shape() == std::vector<int64_t>{2, 2});

    // dA_t = B^T = [[5,7],[6,8]]
    CHECK(grads[0].data()[0] == doctest::Approx(5.f));
    CHECK(grads[0].data()[1] == doctest::Approx(7.f));
    CHECK(grads[0].data()[2] == doctest::Approx(6.f));
    CHECK(grads[0].data()[3] == doctest::Approx(8.f));

    // dB = A_t^T @ grad = [[1,2],[3,4]]
    CHECK(grads[1].data()[0] == doctest::Approx(1.f));
    CHECK(grads[1].data()[1] == doctest::Approx(2.f));
    CHECK(grads[1].data()[2] == doctest::Approx(3.f));
    CHECK(grads[1].data()[3] == doctest::Approx(4.f));
}

TEST_CASE("matmul wires next_edges to input grad_fn") {
    const tensor::Tensor<float> a({2, 2}, std::vector<float>{1.f,1.f,1.f,1.f}, true);
    const tensor::Tensor<float> b({2, 2}, std::vector<float>{2.f,2.f,2.f,2.f}, true);
    const auto A = a.add(b);  // has AddBackward grad_fn
    const tensor::Tensor<float> B({2, 2}, std::vector<float>{1.f,0.f,0.f,1.f}, false);
    const auto C = ops::matmul(A, B);

    REQUIRE(C.grad_fn().get() != nullptr);
    REQUIRE(C.grad_fn()->next_edges.size() == 2);
    CHECK(C.grad_fn()->next_edges[0].get() == A.grad_fn().get());
}
