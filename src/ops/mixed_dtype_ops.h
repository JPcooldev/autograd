#pragma once

// Mixed-dtype binary operations.
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
// The enable_if guard ensures these overloads are only considered when T1 != T2,
// so they never compete with the same-type overloads in elementwise_ops.h.

#include <type_traits>

#include "elementwise_ops.h"

namespace ops {

// ----- add -----

template <typename T1, typename T2,
          typename R = std::common_type_t<T1, T2>,
          std::enable_if_t<!std::is_same<T1, T2>::value, int> = 0>
tensor::Tensor<R> add(const tensor::Tensor<T1>& x, const tensor::Tensor<T2>& y) {
    return ops::add<R>(x.template to<R>(), y.template to<R>());
}

// ----- subtract -----

template <typename T1, typename T2,
          typename R = std::common_type_t<T1, T2>,
          std::enable_if_t<!std::is_same<T1, T2>::value, int> = 0>
tensor::Tensor<R> subtract(const tensor::Tensor<T1>& x, const tensor::Tensor<T2>& y) {
    return ops::subtract<R>(x.template to<R>(), y.template to<R>());
}

// ----- multiply -----

template <typename T1, typename T2,
          typename R = std::common_type_t<T1, T2>,
          std::enable_if_t<!std::is_same<T1, T2>::value, int> = 0>
tensor::Tensor<R> multiply(const tensor::Tensor<T1>& x, const tensor::Tensor<T2>& y) {
    return ops::multiply<R>(x.template to<R>(), y.template to<R>());
}

// ----- divide -----

template <typename T1, typename T2,
          typename R = std::common_type_t<T1, T2>,
          std::enable_if_t<!std::is_same<T1, T2>::value, int> = 0>
tensor::Tensor<R> divide(const tensor::Tensor<T1>& x, const tensor::Tensor<T2>& y) {
    return ops::divide<R>(x.template to<R>(), y.template to<R>());
}

} // namespace ops
