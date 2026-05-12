#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace tensor {

// Lightweight proxy returned by Tensor::operator[].
//
// Each operator[] call advances one dimension.  When all dimensions have been
// consumed (ndim_ == 0) the accessor represents a single element and may be
// implicitly converted to T& (or const T&), or assigned through operator=.
//
// Constness propagates through the chain:
//   Tensor<float>::operator[](i)       -> TensorAccessor<float>
//   Tensor<float>::operator[](i) const -> TensorAccessor<const float>
//   TensorAccessor<float>::operator[](j) const -> TensorAccessor<const float>
//
// The raw pointers into shape_ and strides_ borrow from the owning Tensor.
// The accessor must not outlive the tensor.
template <typename T>
class TensorAccessor {
public:
    TensorAccessor(
        T*             base,
        const int64_t* shape,
        const int64_t* strides,
        int64_t        ndim
    ) : base_(base), shape_(shape), strides_(strides), ndim_(ndim) {}

    // Advance one dimension (mutable).
    TensorAccessor<T> operator[](int64_t idx) {
        bounds_check(idx);
        return TensorAccessor<T>(
            base_    + idx * strides_[0],
            shape_   + 1,
            strides_ + 1,
            ndim_    - 1
        );
    }

    // Advance one dimension (read-only — propagates constness).
    TensorAccessor<const T> operator[](int64_t idx) const {
        bounds_check(idx);
        return TensorAccessor<const T>(
            base_    + idx * strides_[0],
            shape_   + 1,
            strides_ + 1,
            ndim_    - 1
        );
    }

    // Scalar read (mutable).
    operator T&() {
        scalar_check();
        return *base_;
    }

    // Scalar read (const).
    operator const T&() const {
        scalar_check();
        return *base_;
    }

    // Scalar write — disabled when T is const so const chains stay read-only.
    template <typename U = T, typename = std::enable_if_t<!std::is_const<U>::value>>
    TensorAccessor& operator=(const T& val) {
        scalar_check();
        *base_ = val;
        return *this;
    }

private:
    T*             base_;
    const int64_t* shape_;
    const int64_t* strides_;
    int64_t        ndim_;

    void bounds_check(int64_t idx) const {
        if (ndim_ == 0)
            throw std::out_of_range("too many indices: tensor has no remaining dimensions");
        if (idx < 0 || idx >= shape_[0])
            throw std::out_of_range(
                "index " + std::to_string(idx) +
                " is out of bounds for dimension of size " +
                std::to_string(shape_[0]));
    }

    void scalar_check() const {
        if (ndim_ != 0)
            throw std::logic_error(
                "accessor has " + std::to_string(ndim_) +
                " remaining dimension(s); fully index the tensor to get a scalar");
    }
};

} // namespace tensor
