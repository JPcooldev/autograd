#include "../ops/shape_ops.h"
#include "../ops/elementwise_ops.h"
#include "../ops/reduction_ops.h"
#include "../ops/linalg_ops.h"
#include "../autograd/engine.h"

namespace tensor {

// ----- casting -----

/**
 * Casts every element to scalar type U and returns a new packed tensor.
 * Packs first via `contiguous()`, then static_casts each element. Attaches
 * CastBackward when the result is float/double, grad is enabled, and `src` requires grad.
 *
 * @return New `Tensor<U>` with the same shape.
 */
template <typename T>
template <typename U>
Tensor<U> Tensor<T>::to() const {
    const Tensor<T> src = this->contiguous();
    const auto& buf = src.data();
    std::vector<U> new_data(static_cast<size_t>(src.numel()));
    for (size_t i = 0; i < new_data.size(); ++i)
        new_data[i] = static_cast<U>(buf[i]);

    const bool out_is_float = std::is_same<U, float>::value
                           || std::is_same<U, double>::value;
    const bool requires_grad = autograd::is_grad_enabled()
                            && src.requires_grad()
                            && out_is_float;
    std::shared_ptr<autograd::Node<U>> grad_fn = nullptr;
    if (requires_grad)
        grad_fn = std::make_shared<autograd::CastBackward<U, T>>(src);

    return Tensor<U>::from_operation_result(
        src.shape(), std::move(new_data), requires_grad, std::move(grad_fn));
}

/**
 * Casts this tensor to Float32. Wrapper for `to<float>()`.
 *
 * @return New Float32 tensor.
 */
template <typename T>
Tensor<float> Tensor<T>::float32() const {
    return to<float>();
}

/**
 * Casts this tensor to Float64. Wrapper for `to<double>()`.
 *
 * @return New Float64 tensor.
 */
template <typename T>
Tensor<double> Tensor<T>::float64() const {
    return to<double>();
}

/**
 * Casts this tensor to Int32. Wrapper for `to<int32_t>()`.
 *
 * @return New Int32 tensor.
 */
template <typename T>
Tensor<int32_t> Tensor<T>::int32() const {
    return to<int32_t>();
}

/**
 * Casts this tensor to Int64. Wrapper for `to<int64_t>()`.
 *
 * @return New Int64 tensor.
 */
template <typename T>
Tensor<int64_t> Tensor<T>::int64() const {
    return to<int64_t>();
}

// ----- shape operations -----

/**
 * Reinterprets storage as `shape`. Delegates to `ops::reshape`.
 *
 * @param shape Target shape; product of dims must equal `numel()`.
 * @return View with contiguous strides for `shape`.
 *
 * @throws std::invalid_argument if the tensor is a scalar.
 * @throws std::invalid_argument if `numel` would change.
 * @throws std::invalid_argument if any dimension of `shape` is negative.
 * @throws std::overflow_error if computing numel of `shape` overflows int64_t.
 */
template <typename T>
Tensor<T> Tensor<T>::reshape(const std::vector<int64_t>& shape) const {
    return ops::reshape<T>(*this, shape);
}

/**
 * Swaps two dimensions by exchanging shape entries and strides. Delegates to `ops::transpose`.
 *
 * @param dim0 First dimension (negative indices allowed).
 * @param dim1 Second dimension (negative indices allowed).
 * @return View with the two axes exchanged.
 *
 * @throws std::invalid_argument if the tensor is a scalar.
 * @throws std::out_of_range if a dimension is out of bounds.
 */
template <typename T>
Tensor<T> Tensor<T>::transpose(int64_t dim0, int64_t dim1) const {
    return ops::transpose<T>(*this, dim0, dim1);
}

/**
 * Swaps the two axes of a rank-2 tensor. Equivalent to `transpose(0, 1)`.
 *
 * @return View with the two axes exchanged.
 *
 * @throws std::invalid_argument if rank is not 2.
 */
template <typename T>
Tensor<T> Tensor<T>::transpose() const {
    if (this->rank() != 2)
        throw std::invalid_argument("transpose() without dims requires rank-2 tensor");
    return ops::transpose<T>(*this, 0, 1);
}

/**
 * Expands this tensor to `target_shape` by setting stride 0 on broadcast axes.
 * Delegates to `ops::broadcast_to`.
 *
 * @param target_shape Compatible shape (NumPy right-alignment rules).
 * @return View with stride 0 on expanded axes.
 *
 * @throws std::invalid_argument if target rank is less than source rank.
 * @throws std::invalid_argument if a dimension cannot broadcast.
 * @throws std::invalid_argument if any dimension of `target_shape` is negative.
 */
template <typename T>
Tensor<T> Tensor<T>::broadcast_to(const std::vector<int64_t>& target_shape) const {
    return ops::broadcast_to<T>(*this, target_shape);
}

/**
 * Shares storage under a new contiguous shape. Delegates to `ops::view`.
 *
 * @param shape Target shape; product of dims must equal `numel()`.
 * @return View with contiguous strides for `shape`.
 *
 * @throws std::invalid_argument if the tensor is not contiguous.
 * @throws std::invalid_argument if the tensor is a scalar.
 * @throws std::invalid_argument if `numel` would change.
 * @throws std::invalid_argument if any dimension of `shape` is negative.
 */
template <typename T>
Tensor<T> Tensor<T>::view(const std::vector<int64_t>& shape) const {
    return ops::view<T>(*this, shape);
}

/**
 * Collapses dimensions `[start_dim, end_dim]` (inclusive) into one.
 * Delegates to `ops::flatten`.
 *
 * @param start_dim First collapsed axis (negative allowed).
 * @param end_dim Last collapsed axis (negative allowed).
 * @return View with those axes merged.
 *
 * @throws std::invalid_argument if the tensor is a scalar.
 * @throws std::invalid_argument if `start_dim > end_dim` after normalization.
 * @throws std::out_of_range if a dimension is out of bounds.
 */
template <typename T>
Tensor<T> Tensor<T>::flatten(int64_t start_dim, int64_t end_dim) const {
    return ops::flatten<T>(*this, start_dim, end_dim);
}

/**
 * Removes every size-1 dimension. Delegates to `ops::squeeze`.
 *
 * @return View with all size-1 axes dropped.
 */
template <typename T>
Tensor<T> Tensor<T>::squeeze() const {
    return ops::squeeze<T>(*this);
}

/**
 * Removes the size-1 dimension at `dim`. Delegates to `ops::squeeze`.
 *
 * @param dim Axis to drop (negative indices allowed).
 * @return View with that axis removed, or an unchanged view.
 *
 * @throws std::out_of_range if `dim` is out of bounds.
 */
template <typename T>
Tensor<T> Tensor<T>::squeeze(int64_t dim) const {
    return ops::squeeze<T>(*this, dim);
}

/**
 * Inserts a size-1 dimension at `dim`. Delegates to `ops::unsqueeze`.
 *
 * @param dim Insertion index (negative indices allowed).
 * @return View with a new size-1 axis.
 *
 * @throws std::out_of_range if `dim` is outside `[-rank-1, rank]`.
 */
template <typename T>
Tensor<T> Tensor<T>::unsqueeze(int64_t dim) const {
    return ops::unsqueeze<T>(*this, dim);
}

/**
 * Returns a tensor packed into a dense row-major buffer. Delegates to `ops::contiguous`.
 *
 * @return Contiguous tensor with the same shape and logical values.
 */
template <typename T>
Tensor<T> Tensor<T>::contiguous() const {
    return ops::contiguous<T>(*this);
}

/**
 * Returns a view of a slice along `dim`. Delegates to `ops::narrow`.
 *
 * @param dim Axis to slice (negative indices allowed).
 * @param start First index along `dim`.
 * @param length Number of elements to keep.
 * @return View sharing storage with `*this`.
 *
 * @throws std::invalid_argument if the tensor is a scalar.
 * @throws std::invalid_argument if the start/length range is invalid.
 * @throws std::out_of_range if `dim` is out of bounds.
 */
template <typename T>
Tensor<T> Tensor<T>::narrow(int64_t dim, int64_t start, int64_t length) const {
    return ops::narrow<T>(*this, dim, start, length);
}

// ----- elementwise operations -----

/**
 * Element-wise addition. Delegates to `ops::add`.
 *
 * @param other Right-hand operand.
 * @return New tensor `*this + other`.
 *
 * @throws std::invalid_argument on shape mismatch.
 */
template <typename T>
Tensor<T> Tensor<T>::add(const Tensor<T>& other) const {
    return ops::add<T>(*this, other);
}

/**
 * Element-wise negation. Delegates to `ops::neg`.
 *
 * @return New tensor `-(*this)`.
 */
template <typename T>
Tensor<T> Tensor<T>::neg() const {
    return ops::neg<T>(*this);
}

/**
 * Element-wise subtraction. Delegates to `ops::subtract`.
 *
 * @param other Right-hand operand.
 * @return New tensor `*this - other`.
 *
 * @throws std::invalid_argument on shape mismatch.
 */
template <typename T>
Tensor<T> Tensor<T>::subtract(const Tensor<T>& other) const {
    return ops::subtract<T>(*this, other);
}

/**
 * Element-wise multiplication. Delegates to `ops::multiply`.
 *
 * @param other Right-hand operand.
 * @return New tensor `*this * other`.
 *
 * @throws std::invalid_argument on shape mismatch.
 */
template <typename T>
Tensor<T> Tensor<T>::multiply(const Tensor<T>& other) const {
    return ops::multiply<T>(*this, other);
}

/**
 * Element-wise division. Delegates to `ops::divide`.
 *
 * @param other Right-hand operand.
 * @return New tensor `*this / other`.
 *
 * @throws std::invalid_argument on shape mismatch.
 * @throws std::runtime_error on division by zero.
 */
template <typename T>
Tensor<T> Tensor<T>::divide(const Tensor<T>& other) const {
    return ops::divide<T>(*this, other);
}

/**
 * Element-wise power with a scalar exponent. Delegates to `ops::power`.
 *
 * @param exponent Scalar exponent.
 * @return New tensor `(*this) ** exponent`.
 */
template <typename T>
Tensor<T> Tensor<T>::power(const T exponent) const {
    return ops::power<T>(*this, exponent);
}

/**
 * Element-wise absolute value. Delegates to `ops::abs`.
 *
 * @return New tensor `| *this |`.
 */
template <typename T>
Tensor<T> Tensor<T>::abs() const {
    return ops::abs<T>(*this);
}

/**
 * Element-wise exponential. Delegates to `ops::exp`.
 *
 * @return New tensor `exp(*this)`.
 */
template <typename T>
Tensor<T> Tensor<T>::exp() const {
    return ops::exp<T>(*this);
}

/**
 * Element-wise natural logarithm. Delegates to `ops::log`.
 *
 * @return New tensor `log(*this)`.
 *
 * @throws std::runtime_error if any element is non-positive.
 */
template <typename T>
Tensor<T> Tensor<T>::log() const {
    return ops::log<T>(*this);
}

/**
 * Element-wise square root. Delegates to `ops::sqrt`.
 *
 * @return New tensor `sqrt(*this)`.
 */
template <typename T>
Tensor<T> Tensor<T>::sqrt() const {
    return ops::sqrt<T>(*this);
}

/**
 * Element-wise sine. Delegates to `ops::sin`.
 *
 * @return New tensor `sin(*this)`.
 */
template <typename T>
Tensor<T> Tensor<T>::sin() const {
    return ops::sin<T>(*this);
}

/**
 * Element-wise cosine. Delegates to `ops::cos`.
 *
 * @return New tensor `cos(*this)`.
 */
template <typename T>
Tensor<T> Tensor<T>::cos() const {
    return ops::cos<T>(*this);
}

/**
 * Element-wise tangent. Delegates to `ops::tan`.
 *
 * @return New tensor `tan(*this)`.
 */
template <typename T>
Tensor<T> Tensor<T>::tan() const {
    return ops::tan<T>(*this);
}

/**
 * Element-wise hyperbolic sine. Delegates to `ops::sinh`.
 *
 * @return New tensor `sinh(*this)`.
 */
template <typename T>
Tensor<T> Tensor<T>::sinh() const {
    return ops::sinh<T>(*this);
}

/**
 * Element-wise hyperbolic cosine. Delegates to `ops::cosh`.
 *
 * @return New tensor `cosh(*this)`.
 */
template <typename T>
Tensor<T> Tensor<T>::cosh() const {
    return ops::cosh<T>(*this);
}

/**
 * Element-wise hyperbolic tangent. Delegates to `ops::tanh`.
 *
 * @return New tensor `tanh(*this)`.
 */
template <typename T>
Tensor<T> Tensor<T>::tanh() const {
    return ops::tanh<T>(*this);
}

/**
 * Element-wise sigmoid. Delegates to `ops::sigmoid`.
 *
 * @return New tensor `sigmoid(*this)`.
 */
template <typename T>
Tensor<T> Tensor<T>::sigmoid() const {
    return ops::sigmoid<T>(*this);
}

/**
 * Element-wise ReLU. Delegates to `ops::relu`.
 *
 * @return New tensor `relu(*this)`.
 */
template <typename T>
Tensor<T> Tensor<T>::relu() const {
    return ops::relu<T>(*this);
}

/**
 * Element-wise SiLU. Delegates to `ops::silu`.
 *
 * @return New tensor `silu(*this)`.
 */
template <typename T>
Tensor<T> Tensor<T>::silu() const {
    return ops::silu<T>(*this);
}

/**
 * Element-wise GELU (tanh approximation). Delegates to `ops::gelu`.
 *
 * @return New tensor `gelu(*this)`.
 */
template <typename T>
Tensor<T> Tensor<T>::gelu() const {
    return ops::gelu<T>(*this);
}

// ----- reduction operations -----

/**
 * Sums every element into a scalar tensor. Delegates to `ops::sum`.
 *
 * @return Scalar tensor.
 *
 * @throws std::invalid_argument if the tensor is empty.
 */
template <typename T>
Tensor<T> Tensor<T>::sum() const {
    return ops::sum<T>(*this);
}

/**
 * Sums along one axis. Delegates to `ops::sum`.
 *
 * @param dim Axis to reduce (negative indices allowed).
 * @return Tensor with `dim` dropped.
 *
 * @throws std::invalid_argument if the tensor is a scalar.
 * @throws std::out_of_range if `dim` is out of bounds.
 */
template <typename T>
Tensor<T> Tensor<T>::sum(int64_t dim) const {
    return ops::sum<T>(*this, dim);
}

/**
 * Mean of every element as a scalar tensor. Delegates to `ops::mean`.
 *
 * @return Scalar tensor.
 *
 * @throws std::invalid_argument if the tensor is empty.
 */
template <typename T>
Tensor<T> Tensor<T>::mean() const {
    return ops::mean<T>(*this);
}

/**
 * Mean along one axis. Delegates to `ops::mean`.
 *
 * @param dim Axis to reduce (negative indices allowed).
 * @return Tensor with `dim` dropped.
 *
 * @throws std::invalid_argument if the tensor is a scalar.
 * @throws std::invalid_argument if the reduced dimension has size 0.
 * @throws std::out_of_range if `dim` is out of bounds.
 */
template <typename T>
Tensor<T> Tensor<T>::mean(int64_t dim) const {
    return ops::mean<T>(*this, dim);
}

/**
 * Maximum element as a scalar tensor. Delegates to `ops::max`.
 *
 * @return Scalar tensor.
 *
 * @throws std::invalid_argument if the tensor is empty.
 */
template <typename T>
Tensor<T> Tensor<T>::max() const {
    return ops::max<T>(*this);
}

/**
 * Maximum along one axis. Delegates to `ops::max`.
 *
 * @param dim Axis to reduce (negative indices allowed).
 * @return Tensor with `dim` dropped.
 *
 * @throws std::invalid_argument if the tensor is a scalar.
 * @throws std::invalid_argument if the reduced dimension has size 0.
 * @throws std::out_of_range if `dim` is out of bounds.
 */
template <typename T>
Tensor<T> Tensor<T>::max(int64_t dim) const {
    return ops::max<T>(*this, dim);
}

/**
 * Minimum element as a scalar tensor. Delegates to `ops::min`.
 *
 * @return Scalar tensor.
 *
 * @throws std::invalid_argument if the tensor is empty.
 */
template <typename T>
Tensor<T> Tensor<T>::min() const {
    return ops::min<T>(*this);
}

/**
 * Minimum along one axis. Delegates to `ops::min`.
 *
 * @param dim Axis to reduce (negative indices allowed).
 * @return Tensor with `dim` dropped.
 *
 * @throws std::invalid_argument if the tensor is a scalar.
 * @throws std::invalid_argument if the reduced dimension has size 0.
 * @throws std::out_of_range if `dim` is out of bounds.
 */
template <typename T>
Tensor<T> Tensor<T>::min(int64_t dim) const {
    return ops::min<T>(*this, dim);
}

/**
 * Softmax along `dim`. Delegates to `ops::softmax`.
 *
 * @param dim Axis to normalize (negative indices allowed).
 * @return Tensor of the same shape.
 *
 * @throws std::invalid_argument if the tensor is a scalar.
 * @throws std::invalid_argument if the reduced dimension has size 0.
 * @throws std::out_of_range if `dim` is out of bounds.
 */
template <typename T>
Tensor<T> Tensor<T>::softmax(int64_t dim) const {
    return ops::softmax<T>(*this, dim);
}

// ----- linalg operations -----

/**
 * Inner product of two 1-D tensors. Delegates to `ops::dot`.
 *
 * @param other Other vector.
 * @return Scalar tensor.
 *
 * @throws std::invalid_argument if either input is not 1-D or lengths differ.
 */
template <typename T>
Tensor<T> Tensor<T>::dot(const Tensor<T>& other) const {
    return ops::dot<T>(*this, other);
}

/**
 * Matrix product on the last two dimensions. Delegates to `ops::matmul`.
 *
 * @param other Right-hand tensor.
 * @return New contiguous result tensor.
 *
 * @throws std::invalid_argument if either rank is below 2.
 * @throws std::invalid_argument if inner dims mismatch.
 * @throws std::invalid_argument if batch dims are not broadcast-compatible.
 */
template <typename T>
Tensor<T> Tensor<T>::matmul(const Tensor<T>& other) const {
    return ops::matmul<T>(*this, other);
}

// ----- grad / backward -----

/**
 * Returns the accumulated gradient of a leaf, or nullptr if none is stored.
 * Reads `grad_storage_->tensor`.
 *
 * @return Pointer into `GradStorage`, or nullptr.
 */
template <typename T>
const Tensor<T>* Tensor<T>::grad() const {
    if (!grad_storage_ || !grad_storage_->tensor)
        return nullptr;
    return grad_storage_->tensor.get();
}

/**
 * Adds `g` into this leaf's gradient buffer. Packs `g` then copies or accumulates.
 * No-op if `grad_storage_` is null.
 *
 * @param g Incoming gradient.
 */
template <typename T>
void Tensor<T>::accumulate_grad(const Tensor<T>& g) {
    if (!grad_storage_)
        return;
    const Tensor<T> packed = g.contiguous();
    if (!grad_storage_->tensor)
        grad_storage_->tensor = std::make_shared<Tensor<T>>(
            packed.shape(), packed.data(), false);
    else {
        auto& dst = grad_storage_->tensor->data();
        const auto& src = packed.data();
        for (size_t i = 0; i < dst.size(); ++i)
            dst[i] += src[i];
    }
}

/**
 * Resets the gradient tensor to nullptr while keeping `grad_storage_` alive.
 * Aliases created later continue to share the same GradStorage.
 */
template <typename T>
void Tensor<T>::zero_grad() const {
    if (grad_storage_)
        grad_storage_->tensor = nullptr;
}

/**
 * Runs reverse-mode autodiff from this scalar output.
 * Seeds the graph with 1 and calls `autograd::run_backward`.
 *
 * @return `*this` (unchanged).
 *
 * @throws std::invalid_argument if `numel() != 1`.
 * @throws std::invalid_argument if `grad_fn_` is null (leaf).
 */
template <typename T>
Tensor<T> Tensor<T>::backward() const {
    if (numel() != 1)
        throw std::invalid_argument(
            "backward() requires a scalar (numel == 1) output tensor");
    if (!grad_fn_)
        throw std::invalid_argument(
            "backward() called on a leaf tensor; nothing to differentiate");
    const Tensor<T> seed({}, std::vector<T>{T{1}}, false);
    autograd::run_backward(grad_fn_, seed);
    return *this;
}

} // namespace tensor
