#pragma once

// Mixed-dtype two-tensor operations.
//
// When two tensors have different element types, the lower-precision type is
// promoted to the higher-precision one following C's usual arithmetic
// conversions (via std::common_type_t), and the operation is performed in the
// promoted type.
//
// Promotion table (std::common_type_t<T1, T2>):
//
//   T1        | T2        | Result
//   ----------+-----------+----------
//   float     | double    | double
//   int32_t   | int64_t   | int64_t
//   int32_t   | float     | float
//   int32_t   | double    | double
//   int64_t   | float     | float     (*)
//   int64_t   | double    | double
//
// (*) int64_t -> float may lose precision for large integers since float has
//     only 24 bits of mantissa. Prefer casting to double in that case.
//
// Wrapped ops (T1 != T2 overloads in this header):
//   elementwise: add, subtract, multiply, divide
//   linalg:      dot, matmul
//   loss:        l1_loss, l2_loss, mse_loss, nll_loss, cross_entropy_loss,
//                bce_loss, bce_with_logits_loss, kl_div_loss
//   conv:        conv1d, conv2d, conv3d,
//                conv_transpose1d, conv_transpose2d, conv_transpose3d
//
// Not wrapped (not mixed-dtype ops):
//   unary / reductions / pool / dropout / views
//   power          — scalar exponent is already a separate type U
//   cat            — vector<Tensor<T>> cannot hold mixed types; caller .to<R>()
//   Tensor members — stay same-T; mixed is ops:: only
//   Module<T>      — a module is one dtype
//   conv_nd / conv_transpose_nd — internals; use the 1d/2d/3d wrappers
//
// The enable_if guard ensures these overloads are only considered when T1 != T2,
// so they never compete with the same-type overloads.
//
// To add mixed-dtype support for another two-tensor op, keep the same-dtype
// implementation as the source of truth and add a T1 != T2 overload that
// calls mixed_dtype_detail::promote_binary with that op.

#include <type_traits>

#include "elementwise_ops.h"
#include "linalg_ops.h"
#include "loss_ops.h"
#include "conv_ops.h"

namespace ops {
namespace mixed_dtype_detail {

/**
 * Return `x` as a `Tensor<R>`, converting only when the storage type differs.
 *
 * Mechanism: `if constexpr` picks one of two compile-time paths. When `T` is
 * already `R`, returning `x` by value is a shallow Tensor copy (shared
 * storage, same grad_fn). When `T != R`, `.to<R>()` allocates a new buffer
 * and element-casts; autograd is rebuilt on the promoted tensor (the original
 * typed graph is not continued). Skipping `.to<R>()` on the same-type path
 * avoids a needless copy and graph rebuild.
 *
 * `promote_binary` always requests `R = common_type_t<T1, T2>`, so in the
 * mixed-dtype overloads below at least one operand takes the `.to<R>()` path.
 *
 * @param x Operand whose element type may or may not already be `R`.
 * @return `x` if `T == R`, otherwise a new `Tensor<R>` from `.to<R>()`.
 */
template <typename R, typename T>
tensor::Tensor<R> maybe_promote(const tensor::Tensor<T>& x) {
    if constexpr (std::is_same<T, R>::value)
        return x;
    else
        return x.template to<R>();
}

/**
 * Run a same-dtype binary op after promoting both operands to a common type.
 *
 * Mechanism: `R` is `std::common_type_t<T1, T2>` (C usual arithmetic
 * conversions). Each operand is passed through `maybe_promote<R>`, then `op`
 * is invoked on the two `Tensor<R>` values. `op` must be a same-dtype
 * function (or lambda that calls one), e.g. `ops::add`, so kernels, shape
 * checks, and backward nodes stay defined once in the same-dtype headers.
 * Extra arguments (`reduction`, `dim`, conv hyperparameters, bias pointer)
 * belong in the lambda capture, not in this helper.
 *
 * Cross-dtype autograd is not first-class: gradients flow through the
 * promoted `Tensor<R>` copies, not the original `Tensor<T1>` / `Tensor<T2>`.
 *
 * @param x Left operand.
 * @param y Right operand.
 * @param op Callable `(const Tensor<R>&, const Tensor<R>&) -> Tensor<R>`.
 * @return Whatever `op` returns (a `Tensor<R>` for the overloads in this header).
 *
 * @throws whatever `op` throws after promotion (shape mismatch, divide by
 *         zero, …). This helper does not throw on its own.
 */
template <typename T1, typename T2, typename Op>
auto promote_binary(
    const tensor::Tensor<T1>& x,
    const tensor::Tensor<T2>& y,
    Op op
) {
    using R = std::common_type_t<T1, T2>;
    return op(maybe_promote<R>(x), maybe_promote<R>(y));
}

} // namespace mixed_dtype_detail

// ----- elementwise -----

/**
 * Add two tensors of different dtypes. Promotes both to
 * `std::common_type_t<T1, T2>` then calls same-dtype `add`.
 *
 * @param x Left operand.
 * @param y Right operand.
 * @return A tensor in the promoted dtype with the same shape as `x` (and `y`).
 *
 * @throws std::invalid_argument if the tensors have different shapes.
 */
template <typename T1, typename T2,
          typename R = std::common_type_t<T1, T2>,
          std::enable_if_t<!std::is_same<T1, T2>::value, int> = 0>
tensor::Tensor<R> add(const tensor::Tensor<T1>& x, const tensor::Tensor<T2>& y) {
    return mixed_dtype_detail::promote_binary(x, y, [](const auto& a, const auto& b) {
        return ops::add(a, b);
    });
}

/**
 * Subtract two tensors of different dtypes. Promotes both to
 * `std::common_type_t<T1, T2>` then calls same-dtype `subtract`.
 *
 * @param x Left operand.
 * @param y Right operand.
 * @return A tensor in the promoted dtype with the same shape as `x` (and `y`).
 *
 * @throws std::invalid_argument if the tensors have different shapes.
 */
template <typename T1, typename T2,
          typename R = std::common_type_t<T1, T2>,
          std::enable_if_t<!std::is_same<T1, T2>::value, int> = 0>
tensor::Tensor<R> subtract(const tensor::Tensor<T1>& x, const tensor::Tensor<T2>& y) {
    return mixed_dtype_detail::promote_binary(x, y, [](const auto& a, const auto& b) {
        return ops::subtract(a, b);
    });
}

/**
 * Multiply two tensors of different dtypes. Promotes both to
 * `std::common_type_t<T1, T2>` then calls same-dtype `multiply`.
 *
 * @param x Left operand.
 * @param y Right operand.
 * @return A tensor in the promoted dtype with the same shape as `x` (and `y`).
 *
 * @throws std::invalid_argument if the tensors have different shapes.
 */
template <typename T1, typename T2,
          typename R = std::common_type_t<T1, T2>,
          std::enable_if_t<!std::is_same<T1, T2>::value, int> = 0>
tensor::Tensor<R> multiply(const tensor::Tensor<T1>& x, const tensor::Tensor<T2>& y) {
    return mixed_dtype_detail::promote_binary(x, y, [](const auto& a, const auto& b) {
        return ops::multiply(a, b);
    });
}

/**
 * Divide two tensors of different dtypes. Promotes both to
 * `std::common_type_t<T1, T2>` then calls same-dtype `divide`.
 *
 * @param x Numerator.
 * @param y Denominator.
 * @return A tensor in the promoted dtype with the same shape as `x` (and `y`).
 *
 * @throws std::invalid_argument if the tensors have different shapes.
 * @throws std::runtime_error if any element of `y` is zero after promotion.
 */
template <typename T1, typename T2,
          typename R = std::common_type_t<T1, T2>,
          std::enable_if_t<!std::is_same<T1, T2>::value, int> = 0>
tensor::Tensor<R> divide(const tensor::Tensor<T1>& x, const tensor::Tensor<T2>& y) {
    return mixed_dtype_detail::promote_binary(x, y, [](const auto& a, const auto& b) {
        return ops::divide(a, b);
    });
}

// ----- linalg -----

/**
 * Dot product of two 1-D tensors of different dtypes. Promotes both to
 * `std::common_type_t<T1, T2>` then calls same-dtype `dot`.
 *
 * @param x First 1-D tensor.
 * @param y Second 1-D tensor, same length as `x`.
 * @return A scalar tensor of shape `{}` in the promoted dtype.
 *
 * @throws std::invalid_argument if either tensor is not 1-D.
 * @throws std::invalid_argument if the tensors have different lengths.
 */
template <typename T1, typename T2,
          typename R = std::common_type_t<T1, T2>,
          std::enable_if_t<!std::is_same<T1, T2>::value, int> = 0>
tensor::Tensor<R> dot(const tensor::Tensor<T1>& x, const tensor::Tensor<T2>& y) {
    return mixed_dtype_detail::promote_binary(x, y, [](const auto& a, const auto& b) {
        return ops::dot(a, b);
    });
}

/**
 * Matrix product of two tensors of different dtypes. Promotes both to
 * `std::common_type_t<T1, T2>` then calls same-dtype `matmul`.
 *
 * @param A Left operand of rank at least 2.
 * @param B Right operand of rank at least 2.
 * @return The batched matrix product in the promoted dtype.
 *
 * @throws std::invalid_argument if either argument has rank below 2.
 * @throws std::invalid_argument if the inner dimensions `K` do not match.
 * @throws std::invalid_argument if the batch shapes are not broadcastable.
 */
template <typename T1, typename T2,
          typename R = std::common_type_t<T1, T2>,
          std::enable_if_t<!std::is_same<T1, T2>::value, int> = 0>
tensor::Tensor<R> matmul(const tensor::Tensor<T1>& A, const tensor::Tensor<T2>& B) {
    return mixed_dtype_detail::promote_binary(A, B, [](const auto& a, const auto& b) {
        return ops::matmul(a, b);
    });
}

// ----- loss -----

/**
 * L1 loss on tensors of different dtypes. Promotes both to
 * `std::common_type_t<T1, T2>` then calls same-dtype `l1_loss`.
 *
 * @param input Predictions.
 * @param target Targets, same shape as `input`.
 * @param reduction If true, divide the sum by `numel`; otherwise return the sum.
 * @return A scalar tensor of shape `{}` in the promoted dtype.
 *
 * @throws std::invalid_argument if `input` and `target` have different shapes.
 */
template <typename T1, typename T2,
          typename R = std::common_type_t<T1, T2>,
          std::enable_if_t<!std::is_same<T1, T2>::value, int> = 0>
tensor::Tensor<R> l1_loss(
    const tensor::Tensor<T1>& input,
    const tensor::Tensor<T2>& target,
    bool reduction = true) {
    return mixed_dtype_detail::promote_binary(input, target,
        [reduction](const auto& a, const auto& b) {
            return ops::l1_loss(a, b, reduction);
        });
}

/**
 * Squared L2 loss on tensors of different dtypes. Promotes both to
 * `std::common_type_t<T1, T2>` then calls same-dtype `l2_loss`.
 *
 * @param input Predictions.
 * @param target Targets, same shape as `input`.
 * @param reduction If true, divide the sum by `numel`; otherwise return the sum.
 * @return A scalar tensor of shape `{}` in the promoted dtype.
 *
 * @throws std::invalid_argument if `input` and `target` have different shapes.
 */
template <typename T1, typename T2,
          typename R = std::common_type_t<T1, T2>,
          std::enable_if_t<!std::is_same<T1, T2>::value, int> = 0>
tensor::Tensor<R> l2_loss(
    const tensor::Tensor<T1>& input,
    const tensor::Tensor<T2>& target,
    bool reduction = true) {
    return mixed_dtype_detail::promote_binary(input, target,
        [reduction](const auto& a, const auto& b) {
            return ops::l2_loss(a, b, reduction);
        });
}

/**
 * Mean squared error on tensors of different dtypes. Promotes both to
 * `std::common_type_t<T1, T2>` then calls same-dtype `mse_loss`.
 *
 * @param input Predictions.
 * @param target Targets, same shape as `input`.
 * @param reduction If true, divide the sum by `numel`; otherwise return the sum.
 * @return A scalar tensor of shape `{}` in the promoted dtype.
 *
 * @throws std::invalid_argument if `input` and `target` have different shapes.
 */
template <typename T1, typename T2,
          typename R = std::common_type_t<T1, T2>,
          std::enable_if_t<!std::is_same<T1, T2>::value, int> = 0>
tensor::Tensor<R> mse_loss(
    const tensor::Tensor<T1>& input,
    const tensor::Tensor<T2>& target,
    bool reduction = true) {
    return mixed_dtype_detail::promote_binary(input, target,
        [reduction](const auto& a, const auto& b) {
            return ops::mse_loss(a, b, reduction);
        });
}

/**
 * NLL loss on tensors of different dtypes. Promotes both to
 * `std::common_type_t<T1, T2>` then calls same-dtype `nll_loss`.
 *
 * @param input Log-probabilities, same shape as `target`.
 * @param target Soft labels (or one-hot), same shape as `input`.
 * @param dim Class axis (negative indices allowed; default -1).
 * @return A scalar tensor of shape `{}` in the promoted dtype.
 *
 * @throws std::invalid_argument if `input` and `target` have different shapes.
 * @throws std::invalid_argument if `input` is empty.
 * @throws std::out_of_range if `dim` is out of range.
 */
template <typename T1, typename T2,
          typename R = std::common_type_t<T1, T2>,
          std::enable_if_t<!std::is_same<T1, T2>::value, int> = 0>
tensor::Tensor<R> nll_loss(
    const tensor::Tensor<T1>& input,
    const tensor::Tensor<T2>& target,
    int64_t dim = -1) {
    return mixed_dtype_detail::promote_binary(input, target,
        [dim](const auto& a, const auto& b) {
            return ops::nll_loss(a, b, dim);
        });
}

/**
 * Cross-entropy loss on tensors of different dtypes. Promotes both to
 * `std::common_type_t<T1, T2>` then calls same-dtype `cross_entropy_loss`.
 *
 * @param input Logits, same shape as `target`.
 * @param target Soft labels (probabilities), same shape as `input`.
 * @param dim Class axis (negative indices allowed; default -1).
 * @return A scalar tensor of shape `{}` in the promoted dtype.
 *
 * @throws std::invalid_argument if `input` and `target` have different shapes.
 * @throws std::invalid_argument if `input` is a scalar.
 * @throws std::out_of_range if `dim` is out of range.
 */
template <typename T1, typename T2,
          typename R = std::common_type_t<T1, T2>,
          std::enable_if_t<!std::is_same<T1, T2>::value, int> = 0>
tensor::Tensor<R> cross_entropy_loss(
    const tensor::Tensor<T1>& input,
    const tensor::Tensor<T2>& target,
    int64_t dim = -1) {
    return mixed_dtype_detail::promote_binary(input, target,
        [dim](const auto& a, const auto& b) {
            return ops::cross_entropy_loss(a, b, dim);
        });
}

/**
 * Binary cross-entropy on tensors of different dtypes. Promotes both to
 * `std::common_type_t<T1, T2>` then calls same-dtype `bce_loss`.
 *
 * @param input Probabilities, same shape as `target`.
 * @param target Targets in `[0, 1]`, same shape as `input`.
 * @return A scalar tensor of shape `{}` in the promoted dtype.
 *
 * @throws std::invalid_argument if `input` and `target` have different shapes.
 */
template <typename T1, typename T2,
          typename R = std::common_type_t<T1, T2>,
          std::enable_if_t<!std::is_same<T1, T2>::value, int> = 0>
tensor::Tensor<R> bce_loss(
    const tensor::Tensor<T1>& input,
    const tensor::Tensor<T2>& target) {
    return mixed_dtype_detail::promote_binary(input, target, [](const auto& a, const auto& b) {
        return ops::bce_loss(a, b);
    });
}

/**
 * BCE-with-logits on tensors of different dtypes. Promotes both to
 * `std::common_type_t<T1, T2>` then calls same-dtype `bce_with_logits_loss`.
 *
 * @param input Logits, same shape as `target`.
 * @param target Targets in `[0, 1]`, same shape as `input`.
 * @return A scalar tensor of shape `{}` in the promoted dtype.
 *
 * @throws std::invalid_argument if `input` and `target` have different shapes.
 */
template <typename T1, typename T2,
          typename R = std::common_type_t<T1, T2>,
          std::enable_if_t<!std::is_same<T1, T2>::value, int> = 0>
tensor::Tensor<R> bce_with_logits_loss(
    const tensor::Tensor<T1>& input,
    const tensor::Tensor<T2>& target) {
    return mixed_dtype_detail::promote_binary(input, target, [](const auto& a, const auto& b) {
        return ops::bce_with_logits_loss(a, b);
    });
}

/**
 * KL divergence on tensors of different dtypes. Promotes both to
 * `std::common_type_t<T1, T2>` then calls same-dtype `kl_div_loss`.
 *
 * @param input Log-probabilities, same shape as `target`.
 * @param target Probabilities, same shape as `input`.
 * @return A scalar tensor of shape `{}` in the promoted dtype.
 *
 * @throws std::invalid_argument if `input` and `target` have different shapes.
 */
template <typename T1, typename T2,
          typename R = std::common_type_t<T1, T2>,
          std::enable_if_t<!std::is_same<T1, T2>::value, int> = 0>
tensor::Tensor<R> kl_div_loss(
    const tensor::Tensor<T1>& input,
    const tensor::Tensor<T2>& target) {
    return mixed_dtype_detail::promote_binary(input, target, [](const auto& a, const auto& b) {
        return ops::kl_div_loss(a, b);
    });
}

// ----- conv -----

/**
 * 1D convolution of tensors of different dtypes. Promotes `input` and `weight`
 * to `std::common_type_t<T1, T2>` then calls same-dtype `conv1d`. `bias`, if
 * present, must already be the promoted dtype (caller `.to<R>()` otherwise).
 *
 * @param input Input tensor of shape `(N, C, L)`.
 * @param weight Weight tensor of shape `(Cout, Cin, K)`.
 * @param bias Optional bias of shape `{Cout}` in the promoted dtype, or null.
 * @param stride Spatial stride.
 * @param padding Symmetric spatial padding.
 * @param dilation Kernel dilation.
 * @return Output tensor of shape `(N, Cout, L_out)` in the promoted dtype.
 *
 * @throws std::invalid_argument if input rank is not 3.
 * @throws std::invalid_argument if weight rank does not match input rank.
 * @throws std::invalid_argument if weight `in_channels` does not match input.
 * @throws std::invalid_argument if `bias` is present and is not shape `{out_channels}`.
 * @throws std::invalid_argument if `stride` is not greater than 0.
 * @throws std::invalid_argument if the computed output length is not positive.
 */
template <typename T1, typename T2,
          typename R = std::common_type_t<T1, T2>,
          std::enable_if_t<!std::is_same<T1, T2>::value, int> = 0>
tensor::Tensor<R> conv1d(
    const tensor::Tensor<T1>& input,
    const tensor::Tensor<T2>& weight,
    const tensor::Tensor<R>* bias = nullptr,
    int64_t stride = 1,
    int64_t padding = 0,
    int64_t dilation = 1
) {
    return mixed_dtype_detail::promote_binary(input, weight,
        [bias, stride, padding, dilation](const auto& a, const auto& b) {
            return ops::conv1d(a, b, bias, stride, padding, dilation);
        });
}

/**
 * 2D convolution of tensors of different dtypes. Promotes `input` and `weight`
 * to `std::common_type_t<T1, T2>` then calls same-dtype `conv2d`. `bias`, if
 * present, must already be the promoted dtype (caller `.to<R>()` otherwise).
 *
 * @param input Input tensor of shape `(N, C, H, W)`.
 * @param weight Weight tensor of shape `(Cout, Cin, Kh, Kw)`.
 * @param bias Optional bias of shape `{Cout}` in the promoted dtype, or null.
 * @param stride Spatial stride.
 * @param padding Symmetric spatial padding.
 * @param dilation Kernel dilation.
 * @return Output tensor of shape `(N, Cout, H_out, W_out)` in the promoted dtype.
 *
 * @throws std::invalid_argument if input rank is not 4.
 * @throws std::invalid_argument if weight rank does not match input rank.
 * @throws std::invalid_argument if weight `in_channels` does not match input.
 * @throws std::invalid_argument if `bias` is present and is not shape `{out_channels}`.
 * @throws std::invalid_argument if `stride` is not greater than 0.
 * @throws std::invalid_argument if a computed spatial output size is not positive.
 */
template <typename T1, typename T2,
          typename R = std::common_type_t<T1, T2>,
          std::enable_if_t<!std::is_same<T1, T2>::value, int> = 0>
tensor::Tensor<R> conv2d(
    const tensor::Tensor<T1>& input,
    const tensor::Tensor<T2>& weight,
    const tensor::Tensor<R>* bias = nullptr,
    int64_t stride = 1,
    int64_t padding = 0,
    int64_t dilation = 1
) {
    return mixed_dtype_detail::promote_binary(input, weight,
        [bias, stride, padding, dilation](const auto& a, const auto& b) {
            return ops::conv2d(a, b, bias, stride, padding, dilation);
        });
}

/**
 * 3D convolution of tensors of different dtypes. Promotes `input` and `weight`
 * to `std::common_type_t<T1, T2>` then calls same-dtype `conv3d`. `bias`, if
 * present, must already be the promoted dtype (caller `.to<R>()` otherwise).
 *
 * @param input Input tensor of shape `(N, C, D, H, W)`.
 * @param weight Weight tensor of shape `(Cout, Cin, Kd, Kh, Kw)`.
 * @param bias Optional bias of shape `{Cout}` in the promoted dtype, or null.
 * @param stride Spatial stride.
 * @param padding Symmetric spatial padding.
 * @param dilation Kernel dilation.
 * @return Output tensor of shape `(N, Cout, D_out, H_out, W_out)` in the promoted dtype.
 *
 * @throws std::invalid_argument if input rank is not 5.
 * @throws std::invalid_argument if weight rank does not match input rank.
 * @throws std::invalid_argument if weight `in_channels` does not match input.
 * @throws std::invalid_argument if `bias` is present and is not shape `{out_channels}`.
 * @throws std::invalid_argument if `stride` is not greater than 0.
 * @throws std::invalid_argument if a computed spatial output size is not positive.
 */
template <typename T1, typename T2,
          typename R = std::common_type_t<T1, T2>,
          std::enable_if_t<!std::is_same<T1, T2>::value, int> = 0>
tensor::Tensor<R> conv3d(
    const tensor::Tensor<T1>& input,
    const tensor::Tensor<T2>& weight,
    const tensor::Tensor<R>* bias = nullptr,
    int64_t stride = 1,
    int64_t padding = 0,
    int64_t dilation = 1
) {
    return mixed_dtype_detail::promote_binary(input, weight,
        [bias, stride, padding, dilation](const auto& a, const auto& b) {
            return ops::conv3d(a, b, bias, stride, padding, dilation);
        });
}

/**
 * 1D transposed convolution of tensors of different dtypes. Promotes `input`
 * and `weight` to `std::common_type_t<T1, T2>` then calls same-dtype
 * `conv_transpose1d`. `bias`, if present, must already be the promoted dtype.
 *
 * @param input Input tensor of shape `(N, C, L)`.
 * @param weight Weight tensor of shape `(Cin, Cout, K)`.
 * @param bias Optional bias of shape `{Cout}` in the promoted dtype, or null.
 * @param stride Spatial stride.
 * @param padding Symmetric spatial padding.
 * @param dilation Kernel dilation.
 * @param output_padding Extra spatial output padding.
 * @return Output tensor of shape `(N, Cout, L_out)` in the promoted dtype.
 *
 * @throws std::invalid_argument if input rank is not 3.
 * @throws std::invalid_argument if weight rank does not match input rank.
 * @throws std::invalid_argument if weight `in_channels` does not match input.
 * @throws std::invalid_argument if `bias` is present and is not shape `{out_channels}`.
 * @throws std::invalid_argument if `stride` is not greater than 0.
 */
template <typename T1, typename T2,
          typename R = std::common_type_t<T1, T2>,
          std::enable_if_t<!std::is_same<T1, T2>::value, int> = 0>
tensor::Tensor<R> conv_transpose1d(
    const tensor::Tensor<T1>& input,
    const tensor::Tensor<T2>& weight,
    const tensor::Tensor<R>* bias = nullptr,
    int64_t stride = 1,
    int64_t padding = 0,
    int64_t dilation = 1,
    int64_t output_padding = 0
) {
    return mixed_dtype_detail::promote_binary(input, weight,
        [bias, stride, padding, dilation, output_padding](const auto& a, const auto& b) {
            return ops::conv_transpose1d(a, b, bias, stride, padding, dilation, output_padding);
        });
}

/**
 * 2D transposed convolution of tensors of different dtypes. Promotes `input`
 * and `weight` to `std::common_type_t<T1, T2>` then calls same-dtype
 * `conv_transpose2d`. `bias`, if present, must already be the promoted dtype.
 *
 * @param input Input tensor of shape `(N, C, H, W)`.
 * @param weight Weight tensor of shape `(Cin, Cout, Kh, Kw)`.
 * @param bias Optional bias of shape `{Cout}` in the promoted dtype, or null.
 * @param stride Spatial stride.
 * @param padding Symmetric spatial padding.
 * @param dilation Kernel dilation.
 * @param output_padding Extra spatial output padding.
 * @return Output tensor of shape `(N, Cout, H_out, W_out)` in the promoted dtype.
 *
 * @throws std::invalid_argument if input rank is not 4.
 * @throws std::invalid_argument if weight rank does not match input rank.
 * @throws std::invalid_argument if weight `in_channels` does not match input.
 * @throws std::invalid_argument if `bias` is present and is not shape `{out_channels}`.
 * @throws std::invalid_argument if `stride` is not greater than 0.
 */
template <typename T1, typename T2,
          typename R = std::common_type_t<T1, T2>,
          std::enable_if_t<!std::is_same<T1, T2>::value, int> = 0>
tensor::Tensor<R> conv_transpose2d(
    const tensor::Tensor<T1>& input,
    const tensor::Tensor<T2>& weight,
    const tensor::Tensor<R>* bias = nullptr,
    int64_t stride = 1,
    int64_t padding = 0,
    int64_t dilation = 1,
    int64_t output_padding = 0
) {
    return mixed_dtype_detail::promote_binary(input, weight,
        [bias, stride, padding, dilation, output_padding](const auto& a, const auto& b) {
            return ops::conv_transpose2d(a, b, bias, stride, padding, dilation, output_padding);
        });
}

/**
 * 3D transposed convolution of tensors of different dtypes. Promotes `input`
 * and `weight` to `std::common_type_t<T1, T2>` then calls same-dtype
 * `conv_transpose3d`. `bias`, if present, must already be the promoted dtype.
 *
 * @param input Input tensor of shape `(N, C, D, H, W)`.
 * @param weight Weight tensor of shape `(Cin, Cout, Kd, Kh, Kw)`.
 * @param bias Optional bias of shape `{Cout}` in the promoted dtype, or null.
 * @param stride Spatial stride.
 * @param padding Symmetric spatial padding.
 * @param dilation Kernel dilation.
 * @param output_padding Extra spatial output padding.
 * @return Output tensor of shape `(N, Cout, D_out, H_out, W_out)` in the promoted dtype.
 *
 * @throws std::invalid_argument if input rank is not 5.
 * @throws std::invalid_argument if weight rank does not match input rank.
 * @throws std::invalid_argument if weight `in_channels` does not match input.
 * @throws std::invalid_argument if `bias` is present and is not shape `{out_channels}`.
 * @throws std::invalid_argument if `stride` is not greater than 0.
 */
template <typename T1, typename T2,
          typename R = std::common_type_t<T1, T2>,
          std::enable_if_t<!std::is_same<T1, T2>::value, int> = 0>
tensor::Tensor<R> conv_transpose3d(
    const tensor::Tensor<T1>& input,
    const tensor::Tensor<T2>& weight,
    const tensor::Tensor<R>* bias = nullptr,
    int64_t stride = 1,
    int64_t padding = 0,
    int64_t dilation = 1,
    int64_t output_padding = 0
) {
    return mixed_dtype_detail::promote_binary(input, weight,
        [bias, stride, padding, dilation, output_padding](const auto& a, const auto& b) {
            return ops::conv_transpose3d(a, b, bias, stride, padding, dilation, output_padding);
        });
}

} // namespace ops
