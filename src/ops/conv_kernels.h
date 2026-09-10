#pragma once

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace ops {
namespace conv_detail {

/**
 * Multiply a slice of a vector. Computes the product of `v[begin], ..., v[end-1]`.
 *
 * @param v The vector to multiply.
 * @param begin Inclusive start index.
 * @param end Exclusive end index.
 * @return The product of the slice (1 if the range is empty).
 */
inline int64_t product(const std::vector<int64_t>& v, int64_t begin, int64_t end) {
    int64_t p = 1;
    for (int64_t i = begin; i < end; ++i)
        p *= v[static_cast<size_t>(i)];
    return p;
}

/**
 * Compute C-contiguous strides for a shape. Each stride is the product of the
 * following dimensions.
 *
 * @param shape The tensor shape.
 * @return Row-major contiguous strides for `shape`.
 */
inline std::vector<int64_t> contig_strides(const std::vector<int64_t>& shape) {
    std::vector<int64_t> s(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i)
        s[static_cast<size_t>(i)] = s[static_cast<size_t>(i + 1)] * shape[static_cast<size_t>(i + 1)];
    return s;
}

/**
 * Compute one spatial output size for a direct convolution.
 * Uses `(in + 2 * pad - dil * (k - 1) - 1) / stride + 1`.
 *
 * @param in Input spatial size.
 * @param k Kernel size along this axis.
 * @param stride Stride along this axis.
 * @param pad Symmetric padding along this axis.
 * @param dil Dilation along this axis.
 * @return The output spatial size.
 *
 * @throws std::invalid_argument if `stride` is not greater than 0.
 * @throws std::invalid_argument if the computed output size is not positive.
 */
inline int64_t conv_out_size(int64_t in, int64_t k, int64_t stride, int64_t pad, int64_t dil) {
    const int64_t numer = in + 2 * pad - dil * (k - 1) - 1;
    if (stride <= 0)
        throw std::invalid_argument("conv: stride must be > 0");
    const int64_t out = numer / stride + 1;
    if (out <= 0)
        throw std::invalid_argument("conv: computed output size is non-positive");
    return out;
}

/**
 * Compute one spatial output size for a transposed convolution.
 * Uses `(in - 1) * stride - 2 * pad + dil * (k - 1) + out_pad + 1`.
 *
 * @param in Input spatial size.
 * @param k Kernel size along this axis.
 * @param stride Stride along this axis.
 * @param pad Symmetric padding along this axis.
 * @param dil Dilation along this axis.
 * @param out_pad Extra output padding along this axis.
 * @return The output spatial size.
 *
 * @throws std::invalid_argument if `stride` is not greater than 0.
 */
inline int64_t conv_transpose_out_size(
    int64_t in, int64_t k, int64_t stride, int64_t pad, int64_t dil, int64_t out_pad) {
    if (stride <= 0)
        throw std::invalid_argument("conv_transpose: stride must be > 0");
    return (in - 1) * stride - 2 * pad + dil * (k - 1) + out_pad + 1;
}

/**
 * Convert a flat row-major index into multi-dimensional coordinates.
 * Writes one coordinate per dimension of `shape` into `idx`.
 *
 * @param flat The flat index.
 * @param shape The shape to unravel against.
 * @param idx Destination for the coordinates (resized to `shape.size()`).
 */
inline void unravel(int64_t flat, const std::vector<int64_t>& shape, std::vector<int64_t>& idx) {
    idx.resize(shape.size());
    for (int64_t d = static_cast<int64_t>(shape.size()) - 1; d >= 0; --d) {
        idx[static_cast<size_t>(d)] = flat % shape[static_cast<size_t>(d)];
        flat /= shape[static_cast<size_t>(d)];
    }
}

/**
 * Run a direct convolution on packed contiguous buffers.
 * Input is `(N, Cin, *S)`, weight `(Cout, Cin, *K)`, optional bias `(Cout)`,
 * output `(N, Cout, *O)`.
 *
 * @param input Packed input buffer.
 * @param in_shape Input shape `(N, Cin, *S)`.
 * @param weight Packed weight buffer.
 * @param w_shape Weight shape `(Cout, Cin, *K)`.
 * @param bias Optional packed bias of length `Cout`, or null.
 * @param output Packed output buffer.
 * @param out_shape Output shape `(N, Cout, *O)`.
 * @param stride Spatial stride (same on every spatial axis).
 * @param padding Symmetric spatial padding.
 * @param dilation Kernel dilation.
 */
template <typename T>
void conv_forward(
    const T* input, const std::vector<int64_t>& in_shape,
    const T* weight, const std::vector<int64_t>& w_shape,
    const T* bias,
    T* output, const std::vector<int64_t>& out_shape,
    int64_t stride, int64_t padding, int64_t dilation) {
    const int64_t D = static_cast<int64_t>(in_shape.size()) - 2;
    const int64_t Cin = in_shape[1];
    (void)in_shape[0];
    const auto in_st = contig_strides(in_shape);
    const auto w_st = contig_strides(w_shape);
    const auto out_st = contig_strides(out_shape);
    const int64_t out_n = product(out_shape, 0, static_cast<int64_t>(out_shape.size()));

    std::vector<int64_t> oidx;
    std::vector<int64_t> kidx(static_cast<size_t>(D));
    for (int64_t f = 0; f < out_n; ++f) {
        unravel(f, out_shape, oidx); // n, oc, *out_spatial
        const int64_t n = oidx[0];
        const int64_t oc = oidx[1];
        T acc = bias ? bias[oc] : T{0};
        for (int64_t ic = 0; ic < Cin; ++ic) {
            const int64_t kvol = product(w_shape, 2, 2 + D);
            for (int64_t kf = 0; kf < kvol; ++kf) {
                int64_t remaining = kf;
                bool inside = true;
                int64_t in_off = n * in_st[0] + ic * in_st[1];
                int64_t w_off = oc * w_st[0] + ic * w_st[1];
                for (int64_t d = D - 1; d >= 0; --d) {
                    const int64_t kd = remaining % w_shape[static_cast<size_t>(2 + d)];
                    remaining /= w_shape[static_cast<size_t>(2 + d)];
                    const int64_t in_d =
                        oidx[static_cast<size_t>(2 + d)] * stride - padding + kd * dilation;
                    if (in_d < 0 || in_d >= in_shape[static_cast<size_t>(2 + d)]) {
                        inside = false;
                        break;
                    }
                    in_off += in_d * in_st[static_cast<size_t>(2 + d)];
                    w_off += kd * w_st[static_cast<size_t>(2 + d)];
                }
                if (inside)
                    acc += input[in_off] * weight[w_off];
            }
        }
        output[f] = acc;
    }
}

/**
 * Run a transposed convolution on packed contiguous buffers.
 * Input is `(N, Cin, *S)`, weight `(Cin, Cout, *K)`, optional bias `(Cout)`.
 * Output is zeroed, then bias is written if present, then input is scattered.
 *
 * @param input Packed input buffer.
 * @param in_shape Input shape `(N, Cin, *S)`.
 * @param weight Packed weight buffer.
 * @param w_shape Weight shape `(Cin, Cout, *K)`.
 * @param bias Optional packed bias of length `Cout`, or null.
 * @param output Packed output buffer.
 * @param out_shape Output shape `(N, Cout, *O)`.
 * @param stride Spatial stride (same on every spatial axis).
 * @param padding Symmetric spatial padding.
 * @param dilation Kernel dilation.
 */
template <typename T>
void conv_transpose_forward(
    const T* input, const std::vector<int64_t>& in_shape,
    const T* weight, const std::vector<int64_t>& w_shape,
    const T* bias,
    T* output, const std::vector<int64_t>& out_shape,
    int64_t stride, int64_t padding, int64_t dilation) {
    const int64_t D = static_cast<int64_t>(in_shape.size()) - 2;
    const int64_t N = in_shape[0];
    const int64_t Cout = w_shape[1];
    const auto in_st = contig_strides(in_shape);
    const auto w_st = contig_strides(w_shape);
    const auto out_st = contig_strides(out_shape);
    const int64_t out_n = product(out_shape, 0, static_cast<int64_t>(out_shape.size()));

    for (int64_t i = 0; i < out_n; ++i)
        output[i] = T{0};
    if (bias) {
        for (int64_t n = 0; n < N; ++n)
            for (int64_t oc = 0; oc < Cout; ++oc) {
                const int64_t spat = product(out_shape, 2, 2 + D);
                const int64_t base = n * out_st[0] + oc * out_st[1];
                for (int64_t s = 0; s < spat; ++s)
                    output[base + s] = bias[oc];
            }
    }

    const int64_t in_n = product(in_shape, 0, static_cast<int64_t>(in_shape.size()));
    std::vector<int64_t> iidx;
    for (int64_t f = 0; f < in_n; ++f) {
        unravel(f, in_shape, iidx); // n, ic, *in_spatial
        const int64_t n = iidx[0];
        const int64_t ic = iidx[1];
        const T xv = input[f];
        const int64_t kvol = product(w_shape, 2, 2 + D);
        for (int64_t oc = 0; oc < Cout; ++oc) {
            for (int64_t kf = 0; kf < kvol; ++kf) {
                int64_t remaining = kf;
                bool inside = true;
                int64_t out_off = n * out_st[0] + oc * out_st[1];
                int64_t w_off = ic * w_st[0] + oc * w_st[1];
                for (int64_t d = D - 1; d >= 0; --d) {
                    const int64_t kd = remaining % w_shape[static_cast<size_t>(2 + d)];
                    remaining /= w_shape[static_cast<size_t>(2 + d)];
                    const int64_t od =
                        iidx[static_cast<size_t>(2 + d)] * stride - padding + kd * dilation;
                    if (od < 0 || od >= out_shape[static_cast<size_t>(2 + d)]) {
                        inside = false;
                        break;
                    }
                    out_off += od * out_st[static_cast<size_t>(2 + d)];
                    w_off += kd * w_st[static_cast<size_t>(2 + d)];
                }
                if (inside)
                    output[out_off] += xv * weight[w_off];
            }
        }
    }
}

/**
 * Compute the input gradient of a direct convolution.
 * Accumulates a transposed-style scatter of `grad_out` through weight `(Cout, Cin, *K)`.
 *
 * @param grad_out Packed output-gradient buffer.
 * @param out_shape Output shape `(N, Cout, *O)`.
 * @param weight Packed weight buffer.
 * @param w_shape Weight shape `(Cout, Cin, *K)`.
 * @param grad_in Packed input-gradient buffer (zeroed then accumulated).
 * @param in_shape Input shape `(N, Cin, *S)`.
 * @param stride Spatial stride (same on every spatial axis).
 * @param padding Symmetric spatial padding.
 * @param dilation Kernel dilation.
 */
template <typename T>
void conv_backward_input(
    const T* grad_out, const std::vector<int64_t>& out_shape,
    const T* weight, const std::vector<int64_t>& w_shape,
    T* grad_in, const std::vector<int64_t>& in_shape,
    int64_t stride, int64_t padding, int64_t dilation) {
    const int64_t D = static_cast<int64_t>(in_shape.size()) - 2;
    const int64_t Cin = in_shape[1];
    const auto in_st = contig_strides(in_shape);
    const auto w_st = contig_strides(w_shape);
    const auto out_st = contig_strides(out_shape);
    const int64_t in_n = product(in_shape, 0, static_cast<int64_t>(in_shape.size()));
    for (int64_t i = 0; i < in_n; ++i)
        grad_in[i] = T{0};

    const int64_t out_n = product(out_shape, 0, static_cast<int64_t>(out_shape.size()));
    std::vector<int64_t> oidx;
    for (int64_t f = 0; f < out_n; ++f) {
        unravel(f, out_shape, oidx);
        const int64_t n = oidx[0];
        const int64_t oc = oidx[1];
        const T go = grad_out[f];
        for (int64_t ic = 0; ic < Cin; ++ic) {
            const int64_t kvol = product(w_shape, 2, 2 + D);
            for (int64_t kf = 0; kf < kvol; ++kf) {
                int64_t remaining = kf;
                bool inside = true;
                int64_t in_off = n * in_st[0] + ic * in_st[1];
                int64_t w_off = oc * w_st[0] + ic * w_st[1];
                for (int64_t d = D - 1; d >= 0; --d) {
                    const int64_t kd = remaining % w_shape[static_cast<size_t>(2 + d)];
                    remaining /= w_shape[static_cast<size_t>(2 + d)];
                    const int64_t in_d =
                        oidx[static_cast<size_t>(2 + d)] * stride - padding + kd * dilation;
                    if (in_d < 0 || in_d >= in_shape[static_cast<size_t>(2 + d)]) {
                        inside = false;
                        break;
                    }
                    in_off += in_d * in_st[static_cast<size_t>(2 + d)];
                    w_off += kd * w_st[static_cast<size_t>(2 + d)];
                }
                if (inside)
                    grad_in[in_off] += go * weight[w_off];
            }
        }
    }
}

/**
 * Compute the weight gradient of a direct convolution.
 * Accumulates outer products of input windows with `grad_out` into `grad_w`.
 *
 * @param input Packed input buffer.
 * @param in_shape Input shape `(N, Cin, *S)`.
 * @param grad_out Packed output-gradient buffer.
 * @param out_shape Output shape `(N, Cout, *O)`.
 * @param grad_w Packed weight-gradient buffer (zeroed then accumulated).
 * @param w_shape Weight shape `(Cout, Cin, *K)`.
 * @param stride Spatial stride (same on every spatial axis).
 * @param padding Symmetric spatial padding.
 * @param dilation Kernel dilation.
 */
template <typename T>
void conv_backward_weight(
    const T* input, const std::vector<int64_t>& in_shape,
    const T* grad_out, const std::vector<int64_t>& out_shape,
    T* grad_w, const std::vector<int64_t>& w_shape,
    int64_t stride, int64_t padding, int64_t dilation) {
    const int64_t D = static_cast<int64_t>(in_shape.size()) - 2;
    const int64_t w_n = product(w_shape, 0, static_cast<int64_t>(w_shape.size()));
    for (int64_t i = 0; i < w_n; ++i)
        grad_w[i] = T{0};

    const auto in_st = contig_strides(in_shape);
    const auto w_st = contig_strides(w_shape);
    const auto out_st = contig_strides(out_shape);
    const int64_t out_n = product(out_shape, 0, static_cast<int64_t>(out_shape.size()));
    const int64_t Cin = in_shape[1];
    std::vector<int64_t> oidx;
    for (int64_t f = 0; f < out_n; ++f) {
        unravel(f, out_shape, oidx);
        const int64_t n = oidx[0];
        const int64_t oc = oidx[1];
        const T go = grad_out[f];
        for (int64_t ic = 0; ic < Cin; ++ic) {
            const int64_t kvol = product(w_shape, 2, 2 + D);
            for (int64_t kf = 0; kf < kvol; ++kf) {
                int64_t remaining = kf;
                bool inside = true;
                int64_t in_off = n * in_st[0] + ic * in_st[1];
                int64_t w_off = oc * w_st[0] + ic * w_st[1];
                for (int64_t d = D - 1; d >= 0; --d) {
                    const int64_t kd = remaining % w_shape[static_cast<size_t>(2 + d)];
                    remaining /= w_shape[static_cast<size_t>(2 + d)];
                    const int64_t in_d =
                        oidx[static_cast<size_t>(2 + d)] * stride - padding + kd * dilation;
                    if (in_d < 0 || in_d >= in_shape[static_cast<size_t>(2 + d)]) {
                        inside = false;
                        break;
                    }
                    in_off += in_d * in_st[static_cast<size_t>(2 + d)];
                    w_off += kd * w_st[static_cast<size_t>(2 + d)];
                }
                if (inside)
                    grad_w[w_off] += input[in_off] * go;
            }
        }
    }
}

/**
 * Compute the bias gradient of a convolution.
 * Sums `grad_out` over batch and spatial axes for each output channel.
 *
 * @param grad_out Packed output-gradient buffer.
 * @param out_shape Output shape `(N, Cout, *O)`.
 * @param grad_b Packed bias-gradient buffer of length `Cout`.
 */
template <typename T>
void conv_backward_bias(const T* grad_out, const std::vector<int64_t>& out_shape, T* grad_b) {
    const int64_t Cout = out_shape[1];
    const int64_t spat = product(out_shape, 2, static_cast<int64_t>(out_shape.size()));
    const int64_t N = out_shape[0];
    for (int64_t oc = 0; oc < Cout; ++oc)
        grad_b[oc] = T{0};
    const auto st = contig_strides(out_shape);
    for (int64_t n = 0; n < N; ++n)
        for (int64_t oc = 0; oc < Cout; ++oc) {
            const T* row = grad_out + n * st[0] + oc * st[1];
            T s = T{0};
            for (int64_t i = 0; i < spat; ++i)
                s += row[i];
            grad_b[oc] += s;
        }
}

/**
 * Compute the weight gradient of a transposed convolution.
 * Weight layout is `(Cin, Cout, *K)`; accumulates from `(input, grad_out)`.
 *
 * @param input Packed input buffer.
 * @param in_shape Input shape `(N, Cin, *S)`.
 * @param grad_out Packed output-gradient buffer.
 * @param out_shape Output shape `(N, Cout, *O)`.
 * @param grad_w Packed weight-gradient buffer (zeroed then accumulated).
 * @param w_shape Weight shape `(Cin, Cout, *K)`.
 * @param stride Spatial stride (same on every spatial axis).
 * @param padding Symmetric spatial padding.
 * @param dilation Kernel dilation.
 */
template <typename T>
void conv_transpose_backward_weight(
    const T* input, const std::vector<int64_t>& in_shape,
    const T* grad_out, const std::vector<int64_t>& out_shape,
    T* grad_w, const std::vector<int64_t>& w_shape,
    int64_t stride, int64_t padding, int64_t dilation) {
    const int64_t D = static_cast<int64_t>(in_shape.size()) - 2;
    const int64_t w_n = product(w_shape, 0, static_cast<int64_t>(w_shape.size()));
    for (int64_t i = 0; i < w_n; ++i)
        grad_w[i] = T{0};

    const auto in_st = contig_strides(in_shape);
    const auto w_st = contig_strides(w_shape);
    const auto out_st = contig_strides(out_shape);
    const int64_t Cout = w_shape[1];
    const int64_t in_n = product(in_shape, 0, static_cast<int64_t>(in_shape.size()));
    std::vector<int64_t> iidx;
    for (int64_t f = 0; f < in_n; ++f) {
        unravel(f, in_shape, iidx);
        const int64_t n = iidx[0];
        const int64_t ic = iidx[1];
        const T xv = input[f];
        const int64_t kvol = product(w_shape, 2, 2 + D);
        for (int64_t oc = 0; oc < Cout; ++oc) {
            for (int64_t kf = 0; kf < kvol; ++kf) {
                int64_t remaining = kf;
                bool inside = true;
                int64_t out_off = n * out_st[0] + oc * out_st[1];
                int64_t w_off = ic * w_st[0] + oc * w_st[1];
                for (int64_t d = D - 1; d >= 0; --d) {
                    const int64_t kd = remaining % w_shape[static_cast<size_t>(2 + d)];
                    remaining /= w_shape[static_cast<size_t>(2 + d)];
                    const int64_t od =
                        iidx[static_cast<size_t>(2 + d)] * stride - padding + kd * dilation;
                    if (od < 0 || od >= out_shape[static_cast<size_t>(2 + d)]) {
                        inside = false;
                        break;
                    }
                    out_off += od * out_st[static_cast<size_t>(2 + d)];
                    w_off += kd * w_st[static_cast<size_t>(2 + d)];
                }
                if (inside)
                    grad_w[w_off] += xv * grad_out[out_off];
            }
        }
    }
}

/**
 * Compute the input gradient of a transposed convolution.
 * Direct-convolution gather of `grad_out` through weight `(Cin, Cout, *K)`.
 *
 * @param grad_out Packed output-gradient buffer.
 * @param out_shape Output shape `(N, Cout, *O)`.
 * @param weight Packed weight buffer.
 * @param w_shape Weight shape `(Cin, Cout, *K)`.
 * @param grad_in Packed input-gradient buffer.
 * @param in_shape Input shape `(N, Cin, *S)`.
 * @param stride Spatial stride (same on every spatial axis).
 * @param padding Symmetric spatial padding.
 * @param dilation Kernel dilation.
 */
template <typename T>
void conv_transpose_backward_input(
    const T* grad_out, const std::vector<int64_t>& out_shape,
    const T* weight, const std::vector<int64_t>& w_shape,
    T* grad_in, const std::vector<int64_t>& in_shape,
    int64_t stride, int64_t padding, int64_t dilation) {
    const int64_t D = static_cast<int64_t>(in_shape.size()) - 2;
    const int64_t Cout = w_shape[1];
    const auto in_st = contig_strides(in_shape);
    const auto w_st = contig_strides(w_shape);
    const auto out_st = contig_strides(out_shape);
    const int64_t in_n = product(in_shape, 0, static_cast<int64_t>(in_shape.size()));
    for (int64_t i = 0; i < in_n; ++i)
        grad_in[i] = T{0};

    std::vector<int64_t> iidx;
    for (int64_t f = 0; f < in_n; ++f) {
        unravel(f, in_shape, iidx);
        const int64_t n = iidx[0];
        const int64_t ic = iidx[1];
        T acc = T{0};
        const int64_t kvol = product(w_shape, 2, 2 + D);
        for (int64_t oc = 0; oc < Cout; ++oc) {
            for (int64_t kf = 0; kf < kvol; ++kf) {
                int64_t remaining = kf;
                bool inside = true;
                int64_t out_off = n * out_st[0] + oc * out_st[1];
                int64_t w_off = ic * w_st[0] + oc * w_st[1];
                for (int64_t d = D - 1; d >= 0; --d) {
                    const int64_t kd = remaining % w_shape[static_cast<size_t>(2 + d)];
                    remaining /= w_shape[static_cast<size_t>(2 + d)];
                    const int64_t od =
                        iidx[static_cast<size_t>(2 + d)] * stride - padding + kd * dilation;
                    if (od < 0 || od >= out_shape[static_cast<size_t>(2 + d)]) {
                        inside = false;
                        break;
                    }
                    out_off += od * out_st[static_cast<size_t>(2 + d)];
                    w_off += kd * w_st[static_cast<size_t>(2 + d)];
                }
                if (inside)
                    acc += grad_out[out_off] * weight[w_off];
            }
        }
        grad_in[f] = acc;
    }
}

} // namespace conv_detail
} // namespace ops
