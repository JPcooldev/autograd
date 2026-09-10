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
private:
    T*             base_;
    const int64_t* shape_;
    const int64_t* strides_;
    int64_t        ndim_;

    /**
     * Rejects an index that is out of range for the current leading dimension.
     * Used by both const and non-const `operator[]`.
     *
     * @param idx Candidate index along the current dimension.
     *
     * @throws std::out_of_range if `ndim_ == 0` (no remaining dimensions).
     * @throws std::out_of_range if `idx` is outside `[0, shape_[0])`.
     */
    void bounds_check(int64_t idx) const {
        if (ndim_ == 0)
            throw std::out_of_range("too many indices: tensor has no remaining dimensions");
        if (idx < 0 || idx >= shape_[0])
            throw std::out_of_range(
                "index " + std::to_string(idx) +
                " is out of bounds for dimension of size " +
                std::to_string(shape_[0]));
    }

    /**
     * Requires that this accessor already addresses a single element.
     * Used by the conversion operators and scalar assignment.
     *
     * @throws std::logic_error if `ndim_ != 0`.
     */
    void scalar_check() const {
        if (ndim_ != 0)
            throw std::logic_error(
                "accessor has " + std::to_string(ndim_) +
                " remaining dimension(s); fully index the tensor to get a scalar");
    }

public:
    /**
     * Builds an accessor over a (possibly strided) tensor slice.
     * Pointers are borrowed from the owning tensor and are not copied.
     *
     * @param base Pointer to the first element of this slice in storage.
     * @param shape Pointer to the remaining shape dimensions.
     * @param strides Pointer to the remaining strides (in elements).
     * @param ndim Number of remaining dimensions (`0` means a scalar element).
     */
    TensorAccessor(
        T*             base,
        const int64_t* shape,
        const int64_t* strides,
        int64_t        ndim
    ) : base_(base), shape_(shape), strides_(strides), ndim_(ndim) {}

    /**
     * Advances one dimension and returns an accessor for the selected slice.
     * Updates the base pointer by `idx * strides_[0]` and drops the leading axis.
     *
     * @param idx Index along the current leading dimension.
     * @return Accessor for the remaining dimensions (or a scalar when `ndim_ == 1`).
     *
     * @throws std::out_of_range if there are no remaining dimensions.
     * @throws std::out_of_range if `idx` is outside `[0, shape_[0])`.
     */
    TensorAccessor<T> operator[](int64_t idx) {
        bounds_check(idx);
        return TensorAccessor<T>(
            base_    + idx * strides_[0],
            shape_   + 1,
            strides_ + 1,
            ndim_    - 1
        );
    }

    /**
     * Advances one dimension on a const accessor, propagating constness to T.
     * Same indexing rule as the non-const overload.
     *
     * @param idx Index along the current leading dimension.
     * @return Read-only accessor for the remaining dimensions.
     *
     * @throws std::out_of_range if there are no remaining dimensions.
     * @throws std::out_of_range if `idx` is outside `[0, shape_[0])`.
     */
    TensorAccessor<const T> operator[](int64_t idx) const {
        bounds_check(idx);
        return TensorAccessor<const T>(
            base_    + idx * strides_[0],
            shape_   + 1,
            strides_ + 1,
            ndim_    - 1
        );
    }

    /**
     * Converts a fully indexed accessor to a mutable element reference.
     * Valid only when no dimensions remain.
     *
     * @return Reference to the element at `base_`.
     *
     * @throws std::logic_error if `ndim_ != 0`.
     */
    operator T&() {
        scalar_check();
        return *base_;
    }

    /**
     * Converts a fully indexed accessor to a const element reference.
     * Valid only when no dimensions remain.
     *
     * @return Const reference to the element at `base_`.
     *
     * @throws std::logic_error if `ndim_ != 0`.
     */
    operator const T&() const {
        scalar_check();
        return *base_;
    }

    /**
     * Assigns a scalar through a fully indexed mutable accessor.
     * Disabled when T is const so const index chains stay read-only.
     *
     * @param val Value written to `*base_`.
     * @return `*this` after the write.
     *
     * @throws std::logic_error if `ndim_ != 0`.
     */
    template <typename U = T, typename = std::enable_if_t<!std::is_const<U>::value>>
    TensorAccessor& operator=(const T& val) {
        scalar_check();
        *base_ = val;
        return *this;
    }
};

} // namespace tensor
