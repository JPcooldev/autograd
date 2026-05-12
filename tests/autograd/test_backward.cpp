#include <stdexcept>
#include <vector>

#include "../doctest/doctest.h"

#include "../../src/autograd/autograd.h"

// ─── is_leaf / grad / zero_grad ───────────────────────────────────────────────

TEST_CASE("is_leaf returns true for user-created tensors") {
    const tensor::Tensor<float> x({3}, std::vector<float>{1.f, 2.f, 3.f}, true);
    CHECK(x.is_leaf());
}

TEST_CASE("is_leaf returns false for operation outputs") {
    const tensor::Tensor<float> x({3}, std::vector<float>{1.f, 2.f, 3.f}, true);
    const auto y = x.sum();
    CHECK(!y.is_leaf());
}

TEST_CASE("grad returns nullptr before backward is called") {
    const tensor::Tensor<float> x({3}, std::vector<float>{1.f, 2.f, 3.f}, true);
    CHECK(x.grad() == nullptr);
}

TEST_CASE("grad returns nullptr for no-grad leaf") {
    const tensor::Tensor<float> x({3}, std::vector<float>{1.f, 2.f, 3.f}, false);
    CHECK(x.grad() == nullptr);
}

TEST_CASE("grad returns nullptr for non-leaf tensor") {
    const tensor::Tensor<float> x({3}, std::vector<float>{1.f, 2.f, 3.f}, true);
    const auto y = x.sum();
    y.backward();
    CHECK(y.grad() == nullptr);  // y is not a leaf
}

// ─── backward: error cases ────────────────────────────────────────────────────

TEST_CASE("backward throws on non-scalar tensor") {
    const tensor::Tensor<float> x({3}, std::vector<float>{1.f, 2.f, 3.f}, true);
    const auto y = x.add(x);  // shape {3}, not scalar
    CHECK_THROWS_AS(y.backward(), std::invalid_argument);
}

TEST_CASE("backward throws when called on a leaf tensor") {
    const tensor::Tensor<float> x({1}, std::vector<float>{5.f}, true);
    CHECK_THROWS_AS(x.backward(), std::invalid_argument);
}

// ─── backward: single leaf ────────────────────────────────────────────────────

TEST_CASE("backward: loss = sum(x)  →  x.grad == ones") {
    tensor::Tensor<float> x({4}, std::vector<float>{1.f, 2.f, 3.f, 4.f}, true);
    x.sum().backward();

    REQUIRE(x.grad() != nullptr);
    REQUIRE(x.grad()->shape() == std::vector<int64_t>{4});
    for (size_t i = 0; i < 4; ++i)
        CHECK(x.grad()->data()[i] == doctest::Approx(1.f));
}

TEST_CASE("backward: loss = mean(x)  →  x.grad == 1/n") {
    tensor::Tensor<float> x({4}, std::vector<float>{1.f, 2.f, 3.f, 4.f}, true);
    x.mean().backward();

    REQUIRE(x.grad() != nullptr);
    for (size_t i = 0; i < 4; ++i)
        CHECK(x.grad()->data()[i] == doctest::Approx(0.25f));
}

// ─── backward: two leaves ─────────────────────────────────────────────────────

TEST_CASE("backward: loss = sum(x + y)  →  both grads == ones") {
    tensor::Tensor<float> x({3}, std::vector<float>{1.f, 2.f, 3.f}, true);
    tensor::Tensor<float> y({3}, std::vector<float>{4.f, 5.f, 6.f}, true);
    x.add(y).sum().backward();

    REQUIRE(x.grad() != nullptr);
    REQUIRE(y.grad() != nullptr);
    for (size_t i = 0; i < 3; ++i) {
        CHECK(x.grad()->data()[i] == doctest::Approx(1.f));
        CHECK(y.grad()->data()[i] == doctest::Approx(1.f));
    }
}

TEST_CASE("backward: loss = sum(x - y)  →  x.grad == 1, y.grad == -1") {
    tensor::Tensor<float> x({3}, std::vector<float>{3.f, 3.f, 3.f}, true);
    tensor::Tensor<float> y({3}, std::vector<float>{1.f, 1.f, 1.f}, true);
    x.subtract(y).sum().backward();

    REQUIRE(x.grad() != nullptr);
    REQUIRE(y.grad() != nullptr);
    for (size_t i = 0; i < 3; ++i) {
        CHECK(x.grad()->data()[i] == doctest::Approx(1.f));
        CHECK(y.grad()->data()[i] == doctest::Approx(-1.f));
    }
}

// ─── backward: chain (no-grad inputs don't get a gradient) ───────────────────

TEST_CASE("backward: no-grad leaf receives no gradient") {
    tensor::Tensor<float> x({3}, std::vector<float>{1.f, 2.f, 3.f}, true);
    tensor::Tensor<float> c({3}, std::vector<float>{2.f, 2.f, 2.f}, false); // no grad
    x.multiply(c).sum().backward();

    REQUIRE(x.grad() != nullptr);
    CHECK(c.grad() == nullptr);  // requires_grad=false leaf never gets a gradient
}

// ─── backward: quadratic (shared leaf x used twice) ──────────────────────────

TEST_CASE("backward: loss = sum(x * x)  →  x.grad == 2*x") {
    // d/dx sum(x^2) = 2x
    tensor::Tensor<float> x({3}, std::vector<float>{1.f, 2.f, 3.f}, true);
    x.multiply(x).sum().backward();

    REQUIRE(x.grad() != nullptr);
    CHECK(x.grad()->data()[0] == doctest::Approx(2.f));
    CHECK(x.grad()->data()[1] == doctest::Approx(4.f));
    CHECK(x.grad()->data()[2] == doctest::Approx(6.f));
}

TEST_CASE("backward: loss = sum(x^2)  →  x.grad == 2*x  (via power op)") {
    tensor::Tensor<float> x({3}, std::vector<float>{1.f, 2.f, 3.f}, true);
    x.power(2.f).sum().backward();

    REQUIRE(x.grad() != nullptr);
    CHECK(x.grad()->data()[0] == doctest::Approx(2.f));
    CHECK(x.grad()->data()[1] == doctest::Approx(4.f));
    CHECK(x.grad()->data()[2] == doctest::Approx(6.f));
}

// ─── backward: longer chain ───────────────────────────────────────────────────

TEST_CASE("backward: loss = sum(exp(x))  →  x.grad == exp(x)") {
    tensor::Tensor<float> x({3}, std::vector<float>{0.f, 1.f, 2.f}, true);
    x.exp().sum().backward();

    REQUIRE(x.grad() != nullptr);
    CHECK(x.grad()->data()[0] == doctest::Approx(std::exp(0.f)));
    CHECK(x.grad()->data()[1] == doctest::Approx(std::exp(1.f)));
    CHECK(x.grad()->data()[2] == doctest::Approx(std::exp(2.f)));
}

TEST_CASE("backward: loss = sum(sigmoid(x))  →  x.grad == sigmoid(x)*(1-sigmoid(x))") {
    const std::vector<float> vals{-1.f, 0.f, 1.f};
    tensor::Tensor<float> x({3}, vals, true);
    x.sigmoid().sum().backward();

    REQUIRE(x.grad() != nullptr);
    for (size_t i = 0; i < 3; ++i) {
        const float s = 1.f / (1.f + std::exp(-vals[i]));
        CHECK(x.grad()->data()[i] == doctest::Approx(s * (1.f - s)).epsilon(1e-5));
    }
}

// ─── backward: zero_grad and accumulation ────────────────────────────────────

TEST_CASE("zero_grad resets accumulated gradient to nullptr") {
    tensor::Tensor<float> x({3}, std::vector<float>{1.f, 2.f, 3.f}, true);
    x.sum().backward();
    REQUIRE(x.grad() != nullptr);
    x.zero_grad();
    CHECK(x.grad() == nullptr);
}

TEST_CASE("two backward calls without zero_grad accumulate gradients") {
    // After two calls: grad should be 2 * ones
    tensor::Tensor<float> x({3}, std::vector<float>{1.f, 2.f, 3.f}, true);
    x.sum().backward();
    x.sum().backward();  // second call: grad += ones

    REQUIRE(x.grad() != nullptr);
    for (size_t i = 0; i < 3; ++i)
        CHECK(x.grad()->data()[i] == doctest::Approx(2.f));
}

TEST_CASE("zero_grad then backward gives fresh gradient") {
    tensor::Tensor<float> x({3}, std::vector<float>{1.f, 2.f, 3.f}, true);
    x.sum().backward();
    x.zero_grad();
    x.sum().backward();

    REQUIRE(x.grad() != nullptr);
    for (size_t i = 0; i < 3; ++i)
        CHECK(x.grad()->data()[i] == doctest::Approx(1.f));
}

// ─── backward: linalg ops ─────────────────────────────────────────────────────

TEST_CASE("backward: loss = dot(x, y).sum()  →  x.grad == y, y.grad == x") {
    // dot returns a scalar {}, backward seed is 1
    tensor::Tensor<float> x({3}, std::vector<float>{1.f, 2.f, 3.f}, true);
    tensor::Tensor<float> y({3}, std::vector<float>{4.f, 5.f, 6.f}, true);
    x.dot(y).backward();  // result is already scalar {}

    REQUIRE(x.grad() != nullptr);
    REQUIRE(y.grad() != nullptr);
    CHECK(x.grad()->data()[0] == doctest::Approx(4.f));
    CHECK(x.grad()->data()[1] == doctest::Approx(5.f));
    CHECK(x.grad()->data()[2] == doctest::Approx(6.f));
    CHECK(y.grad()->data()[0] == doctest::Approx(1.f));
    CHECK(y.grad()->data()[1] == doctest::Approx(2.f));
    CHECK(y.grad()->data()[2] == doctest::Approx(3.f));
}

TEST_CASE("backward: loss = sum(A @ B)  →  dA = ones_grad @ B^T, dB = A^T @ ones_grad") {
    // A = [[1,2],[3,4]], B = [[5,6],[7,8]]
    // C = A @ B = [[19,22],[43,50]]
    // loss = sum(C), grad_C = ones(2,2)
    // dA = grad_C @ B^T = [[11,15],[11,15]], dB = A^T @ grad_C = [[4,4],[6,6]]
    tensor::Tensor<float> A({2, 2}, std::vector<float>{1.f, 2.f, 3.f, 4.f}, true);
    tensor::Tensor<float> B({2, 2}, std::vector<float>{5.f, 6.f, 7.f, 8.f}, true);
    A.matmul(B).sum().backward();

    REQUIRE(A.grad() != nullptr);
    REQUIRE(B.grad() != nullptr);

    // dA = [[11,15],[11,15]]
    CHECK(A.grad()->data()[0] == doctest::Approx(11.f));
    CHECK(A.grad()->data()[1] == doctest::Approx(15.f));
    CHECK(A.grad()->data()[2] == doctest::Approx(11.f));
    CHECK(A.grad()->data()[3] == doctest::Approx(15.f));

    // dB = [[4,4],[6,6]]
    CHECK(B.grad()->data()[0] == doctest::Approx(4.f));
    CHECK(B.grad()->data()[1] == doctest::Approx(4.f));
    CHECK(B.grad()->data()[2] == doctest::Approx(6.f));
    CHECK(B.grad()->data()[3] == doctest::Approx(6.f));
}
