/*
 * Autograd engine: leaf vs non-leaf, backward, accumulation, casts, grad mode.
 *
 * - is_leaf on a user tensor vs an op output
 * - .grad() is null before backward, on a no-grad leaf, and on a non-leaf
 * - backward() rejects a non-scalar output and a leaf
 * - sum / mean / add / subtract / multiply / power / exp / sigmoid grads
 * - no-grad operand is skipped; the other leaf still gets the product rule
 * - zero_grad, accumulation across two passes, then a fresh pass
 * - dot and 2-D matmul grads
 * - float/double casts, sandwich casts, mixed-dtype add
 * - NoGradContext: no grad_fn, restore on exit, nested guards, AutoGradContext
 * - diamond (h+h) accumulation, reusing one graph, non-1 upstream scale
 * - 0-dim backward, disconnected leaf stays without a grad
 */

#include <cmath>
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
    CHECK(x.grad()->data()[0] == doctest::Approx(2.f));
    CHECK(x.grad()->data()[1] == doctest::Approx(2.f));
    CHECK(x.grad()->data()[2] == doctest::Approx(2.f));
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

// ─── backward: casting ────────────────────────────────────────────────────────

TEST_CASE("backward: float -> double -> sum  writes float leaf grad") {
    tensor::Tensor<float> x({3}, std::vector<float>{1.f, 2.f, 3.f}, true);
    x.to<double>().sum().backward();

    REQUIRE(x.grad() != nullptr);
    for (size_t i = 0; i < 3; ++i)
        CHECK(x.grad()->data()[i] == doctest::Approx(1.f));
}

TEST_CASE("backward: double -> float -> sum  writes double leaf grad") {
    tensor::Tensor<double> x({2}, std::vector<double>{1.0, 2.0}, true);
    x.float32().sum().backward();

    REQUIRE(x.grad() != nullptr);
    CHECK(x.grad()->data()[0] == doctest::Approx(1.0));
    CHECK(x.grad()->data()[1] == doctest::Approx(1.0));
}

TEST_CASE("backward: same-dtype to<float>() stays on the graph") {
    tensor::Tensor<float> x({2}, std::vector<float>{4.f, 5.f}, true);
    x.to<float>().sum().backward();

    REQUIRE(x.grad() != nullptr);
    CHECK(x.grad()->data()[0] == doctest::Approx(1.f));
    CHECK(x.grad()->data()[1] == doctest::Approx(1.f));
}

TEST_CASE("backward: float -> double -> float sandwich") {
    tensor::Tensor<float> x({2}, std::vector<float>{1.f, 3.f}, true);
    x.to<double>().to<float>().sum().backward();

    REQUIRE(x.grad() != nullptr);
    CHECK(x.grad()->data()[0] == doctest::Approx(1.f));
    CHECK(x.grad()->data()[1] == doctest::Approx(1.f));
}

TEST_CASE("backward: cast after a differentiable op") {
    tensor::Tensor<float> x({2}, std::vector<float>{2.f, 3.f}, true);
    x.add(x).to<double>().sum().backward();

    REQUIRE(x.grad() != nullptr);
    CHECK(x.grad()->data()[0] == doctest::Approx(2.f));
    CHECK(x.grad()->data()[1] == doctest::Approx(2.f));
}

TEST_CASE("backward: mixed-dtype add float + double") {
    tensor::Tensor<float> a({2}, std::vector<float>{1.f, 2.f}, true);
    tensor::Tensor<double> b({2}, std::vector<double>{3.0, 4.0}, true);
    ops::add(a, b).sum().backward();

    REQUIRE(a.grad() != nullptr);
    REQUIRE(b.grad() != nullptr);
    CHECK(a.grad()->data()[0] == doctest::Approx(1.f));
    CHECK(a.grad()->data()[1] == doctest::Approx(1.f));
    CHECK(b.grad()->data()[0] == doctest::Approx(1.0));
    CHECK(b.grad()->data()[1] == doctest::Approx(1.0));
}

TEST_CASE("to() under NoGradContext does not attach grad_fn") {
    tensor::Tensor<float> x({2}, std::vector<float>{1.f, 2.f}, true);
    CHECK(autograd::is_grad_enabled());
    {
        autograd::NoGradContext guard;
        CHECK_FALSE(autograd::is_grad_enabled());
        const auto y = x.to<double>();
        CHECK_FALSE(y.requires_grad());
        CHECK(y.grad_fn().get() == nullptr);
        const auto z = x.add(x);
        CHECK_FALSE(z.requires_grad());
        CHECK(z.grad_fn().get() == nullptr);
    }
    CHECK(autograd::is_grad_enabled());
    const auto w = x.add(x);
    CHECK(w.requires_grad());
    CHECK(w.grad_fn().get() != nullptr);
}

TEST_CASE("nested NoGradContext restores the inner disabled mode") {
    CHECK(autograd::is_grad_enabled());
    {
        autograd::NoGradContext outer;
        CHECK_FALSE(autograd::is_grad_enabled());
        {
            autograd::NoGradContext inner;
            CHECK_FALSE(autograd::is_grad_enabled());
        }
        CHECK_FALSE(autograd::is_grad_enabled());
        autograd::AutoGradContext enable(true);
        CHECK(autograd::is_grad_enabled());
    }
    CHECK(autograd::is_grad_enabled());
}

TEST_CASE("diamond graph accumulates into a non-leaf intermediate") {
    tensor::Tensor<float> x({3}, std::vector<float>{-1.f, 0.f, 2.f}, true);
    auto h = x.relu();
    h.add(h).sum().backward();
    REQUIRE(x.grad() != nullptr);
    CHECK(x.grad()->data()[0] == doctest::Approx(0.f));
    CHECK(x.grad()->data()[1] == doctest::Approx(0.f));
    CHECK(x.grad()->data()[2] == doctest::Approx(2.f));
}

TEST_CASE("reusing one graph accumulates leaf grads") {
    tensor::Tensor<float> x({3}, std::vector<float>{1.f, 2.f, 3.f}, true);
    auto y = x.sum();
    y.backward();
    y.backward();
    REQUIRE(x.grad() != nullptr);
    for (size_t i = 0; i < 3; ++i)
        CHECK(x.grad()->data()[i] == doctest::Approx(2.f));
}

TEST_CASE("non-unit upstream scalar scales the leaf gradient") {
    tensor::Tensor<float> x({2}, std::vector<float>{1.f, 2.f}, true);
    x.sum().exp().backward();
    const float scale = std::exp(3.f);
    REQUIRE(x.grad() != nullptr);
    CHECK(x.grad()->data()[0] == doctest::Approx(scale));
    CHECK(x.grad()->data()[1] == doctest::Approx(scale));
}

TEST_CASE("0-dim tensor backward seeds a matching scalar") {
    tensor::Tensor<float> x({}, std::vector<float>{2.f}, true);
    x.exp().backward();
    REQUIRE(x.grad() != nullptr);
    CHECK(x.grad()->shape() == std::vector<int64_t>{});
    CHECK(x.grad()->data()[0] == doctest::Approx(std::exp(2.f)));
}

TEST_CASE("disconnected leaf is left without a gradient") {
    tensor::Tensor<float> x({2}, std::vector<float>{1.f, 2.f}, true);
    tensor::Tensor<float> y({2}, std::vector<float>{3.f, 4.f}, true);
    x.sum().backward();
    REQUIRE(x.grad() != nullptr);
    CHECK(y.grad() == nullptr);
}
