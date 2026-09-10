#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "autograd/autograd.h"
#include "logging/logger.h"

namespace {

std::string fmt(float v) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(4) << v;
    return oss.str();
}

/**
 * Format a rank-2 tensor as a nested list for logging.
 *
 * @param t Rank-2 tensor to print.
 * @param name Label prefixed onto the formatted string.
 * @return A single-line string such as `C (2x3) = [[...], [...]]`.
 */
std::string format_matrix(const tensor::Tensor<float>& t, const std::string& name) {
    const auto& shape = t.shape();
    std::ostringstream oss;
    oss << name << " (" << shape[0] << "x" << shape[1] << ") = [";
    for (int64_t i = 0; i < shape[0]; ++i) {
        if (i > 0)
            oss << ", ";
        oss << "[";
        for (int64_t j = 0; j < shape[1]; ++j) {
            if (j > 0)
                oss << ", ";
            oss << fmt(static_cast<float>(t[i][j]));
        }
        oss << "]";
    }
    oss << "]";
    return oss.str();
}

} // namespace

int main() {
    logging::LoggingContext logs(true);

    logging::info("matmul example: C = A @ B, then d(sum(C))/dA via autograd");

    tensor::Tensor<float> A({2, 3}, {1.f, 2.f, 3.f, 4.f, 5.f, 6.f}, true);
    tensor::Tensor<float> B({2, 3}, {0.5f, 1.f, 1.5f, 2.f, 2.5f, 3.f}, false);

    logging::info(format_matrix(A, "A"));
    logging::info(format_matrix(B, "B"));
    logging::info("computing C = ops::matmul(A, B.transpose())");
    auto C = ops::matmul(A, B.transpose());
    logging::info(format_matrix(C, "C"));

    logging::info("running backward on sum(C) to fill A.grad");
    auto loss = C.sum();
    loss.backward();

    const tensor::Tensor<float>* gA = A.grad();
    logging::info("sum(C) = " + fmt(loss.data()[0]));
    logging::info(format_matrix(*gA, "A.grad"));

    std::cout << "\n========== FINAL RESULTS ==========\n"
              << format_matrix(C, "C") << "\n"
              << "sum(C) = " << fmt(loss.data()[0]) << "\n"
              << format_matrix(*gA, "A.grad") << "\n"
              << "===================================\n";
    return 0;
}
