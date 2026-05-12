#include <cstdint>
#include <stdexcept>
#include <vector>

#include "../doctest/doctest.h"
#include "../../src/tensor/tensor.h"

// ============================================================
//  Read access
// ============================================================

TEST_CASE("operator[] reads correct element from 1D tensor") {
    const tensor::Tensor<float> t({4}, {10.f, 20.f, 30.f, 40.f}, false);
    CHECK(static_cast<float>(t[0]) == doctest::Approx(10.f));
    CHECK(static_cast<float>(t[3]) == doctest::Approx(40.f));
}

TEST_CASE("operator[] reads correct element from 2D tensor") {
    // shape [2,3]: [[1,2,3],[4,5,6]]
    const tensor::Tensor<float> t({2, 3}, {1.f, 2.f, 3.f, 4.f, 5.f, 6.f}, false);
    CHECK(static_cast<float>(t[0][0]) == doctest::Approx(1.f));
    CHECK(static_cast<float>(t[0][2]) == doctest::Approx(3.f));
    CHECK(static_cast<float>(t[1][0]) == doctest::Approx(4.f));
    CHECK(static_cast<float>(t[1][2]) == doctest::Approx(6.f));
}

TEST_CASE("operator[] reads correct element from 3D tensor") {
    // shape [2,3,4]: row-major sequential values 0..23
    std::vector<float> values(24);
    for (int i = 0; i < 24; ++i) values[i] = static_cast<float>(i);

    const tensor::Tensor<float> t({2, 3, 4}, values, false);

    // strides: [12, 4, 1]
    // t[i][j][k] == i*12 + j*4 + k
    CHECK(static_cast<float>(t[0][0][0]) == doctest::Approx(0.f));
    CHECK(static_cast<float>(t[0][1][2]) == doctest::Approx(6.f));
    CHECK(static_cast<float>(t[1][2][3]) == doctest::Approx(23.f));
    CHECK(static_cast<float>(t[1][0][1]) == doctest::Approx(13.f));
}

TEST_CASE("const operator[] chain is read-only (const tensor)") {
    const tensor::Tensor<float> t({2, 2}, {1.f, 2.f, 3.f, 4.f}, false);
    const float v = t[1][0];
    CHECK(v == doctest::Approx(3.f));
}

// ============================================================
//  Write access
// ============================================================

TEST_CASE("operator[] allows element assignment in 1D tensor") {
    tensor::Tensor<float> t({3}, {0.f, 0.f, 0.f}, false);
    t[1] = 99.f;
    CHECK(static_cast<float>(t[1]) == doctest::Approx(99.f));
    CHECK(static_cast<float>(t[0]) == doctest::Approx(0.f));
    CHECK(static_cast<float>(t[2]) == doctest::Approx(0.f));
}

TEST_CASE("operator[] allows element assignment in 2D tensor") {
    tensor::Tensor<float> t({2, 3}, {0.f, 0.f, 0.f, 0.f, 0.f, 0.f}, false);
    t[1][2] = 7.f;
    CHECK(static_cast<float>(t[1][2]) == doctest::Approx(7.f));
    CHECK(static_cast<float>(t[0][0]) == doctest::Approx(0.f));
}

TEST_CASE("operator[] assignment modifies underlying storage") {
    tensor::Tensor<float> t({2, 2}, {1.f, 2.f, 3.f, 4.f}, false);
    t[0][1] = 99.f;
    CHECK(t.data()[1] == doctest::Approx(99.f));
}

// ============================================================
//  Bounds checking
// ============================================================

TEST_CASE("operator[] throws on out-of-range index") {
    tensor::Tensor<float> t({3, 4}, {}, false);
    (void)t; // shape-only

    // Re-create properly
    tensor::Tensor<float> t2({3, 4}, false);
    CHECK_THROWS_AS(t2[3],    std::out_of_range);  // dim 0 has size 3
    CHECK_THROWS_AS(t2[-1],   std::out_of_range);
    CHECK_THROWS_AS(t2[0][4], std::out_of_range);  // dim 1 has size 4
}

TEST_CASE("operator[] throws when too many indices applied") {
    tensor::Tensor<float> t({2, 2}, false);
    auto row = t[0];
    auto elem = row[1];
    CHECK_THROWS_AS(elem[0], std::out_of_range);  // ndim_ is 0 now
}

// ============================================================
//  Non-float types
// ============================================================

TEST_CASE("operator[] works for int32_t tensor") {
    tensor::Tensor<int32_t> t({2, 2}, {1, 2, 3, 4}, false);
    CHECK(static_cast<int32_t>(t[0][0]) == 1);
    CHECK(static_cast<int32_t>(t[1][1]) == 4);
    t[0][1] = 99;
    CHECK(static_cast<int32_t>(t[0][1]) == 99);
}
