#include "../ops/shape_ops.h"
#include "../ops/elementwise_ops.h"
#include "../ops/reduction_ops.h"
#include "../ops/linalg_ops.h"
#include "../autograd/engine.h"

namespace tensor {

// ----- casting -----

template <typename T>
template <typename U>
Tensor<U> Tensor<T>::to() const {
    const auto& src = data();
    std::vector<U> new_data(src.size());
    for (size_t i = 0; i < new_data.size(); ++i)
        new_data[i] = static_cast<U>(src[i]);
    return Tensor<U>::from_operation_result(shape_, std::move(new_data), false, nullptr);
}

// ----- shape operations -----

template <typename T>
Tensor<T> Tensor<T>::reshape(const std::vector<int64_t>& shape) const {
    return ops::reshape<T>(*this, shape);
}

template <typename T>
Tensor<T> Tensor<T>::transpose(
    int64_t dim0, 
    int64_t dim1
) const {
    return ops::transpose<T>(*this, dim0, dim1);
}

template <typename T>
Tensor<T> Tensor<T>::transpose() const {
    if (this->rank() != 2) {
        throw std::invalid_argument("transpose() without dims requires rank-2 tensor");
    }
    return ops::transpose<T>(*this, 0, 1);
}

template <typename T>
Tensor<T> Tensor<T>::broadcast_to(const std::vector<int64_t>& target_shape) const {
    return ops::broadcast_to<T>(*this, target_shape);
}

template <typename T>
Tensor<T> Tensor<T>::view(const std::vector<int64_t>& shape) const {
    return ops::view<T>(*this, shape);
}

template <typename T>
Tensor<T> Tensor<T>::flatten(int64_t start_dim, int64_t end_dim) const {
    return ops::flatten<T>(*this, start_dim, end_dim);
}

template <typename T>
Tensor<T> Tensor<T>::squeeze() const {
    return ops::squeeze<T>(*this);
}

template <typename T>
Tensor<T> Tensor<T>::squeeze(int64_t dim) const {
    return ops::squeeze<T>(*this, dim);
}

template <typename T>
Tensor<T> Tensor<T>::unsqueeze(int64_t dim) const {
    return ops::unsqueeze<T>(*this, dim);
}

// ----- elementwise operations -----

template <typename T>
Tensor<T> Tensor<T>::add(const Tensor<T>& other) const {
    return ops::add<T>(*this, other);
}

template <typename T>
Tensor<T> Tensor<T>::neg() const {
    return ops::neg<T>(*this);
}

template <typename T>
Tensor<T> Tensor<T>::subtract(const Tensor<T>& other) const {
    return ops::subtract<T>(*this, other);
}

template <typename T>
Tensor<T> Tensor<T>::multiply(const Tensor<T>& other) const {
    return ops::multiply<T>(*this, other);
}

template <typename T>
Tensor<T> Tensor<T>::divide(const Tensor<T>& other) const {
    return ops::divide<T>(*this, other);
}

template <typename T>
Tensor<T> Tensor<T>::power(const T exponent) const {
    return ops::power<T>(*this, exponent);
}

template <typename T>
Tensor<T> Tensor<T>::abs() const {
    return ops::abs<T>(*this);
}

template <typename T>
Tensor<T> Tensor<T>::exp() const {
    return ops::exp<T>(*this);
}

template <typename T>
Tensor<T> Tensor<T>::log() const {
    return ops::log<T>(*this);
}

template <typename T>
Tensor<T> Tensor<T>::sin() const {
    return ops::sin<T>(*this);
}

template <typename T>
Tensor<T> Tensor<T>::cos() const {
    return ops::cos<T>(*this);
}

template <typename T>
Tensor<T> Tensor<T>::tan() const {
    return ops::tan<T>(*this);
}

template <typename T>
Tensor<T> Tensor<T>::sinh() const {
    return ops::sinh<T>(*this);
}

template <typename T>
Tensor<T> Tensor<T>::cosh() const {
    return ops::cosh<T>(*this);
}

template <typename T>
Tensor<T> Tensor<T>::tanh() const {
    return ops::tanh<T>(*this);
}

template <typename T>
Tensor<T> Tensor<T>::sigmoid() const {
    return ops::sigmoid<T>(*this);
}

template <typename T>
Tensor<T> Tensor<T>::relu() const {
    return ops::relu<T>(*this);
}

template <typename T>
Tensor<T> Tensor<T>::silu() const {
    return ops::silu<T>(*this);
}

template <typename T>
Tensor<T> Tensor<T>::gelu() const {
    return ops::gelu<T>(*this);
}

// ----- reduction operations -----

template <typename T>
Tensor<T> Tensor<T>::sum() const {
    return ops::sum<T>(*this);
}

template <typename T>
Tensor<T> Tensor<T>::sum(int64_t dim) const {
    return ops::sum<T>(*this, dim);
}

template <typename T>
Tensor<T> Tensor<T>::mean() const {
    return ops::mean<T>(*this);
}

template <typename T>
Tensor<T> Tensor<T>::mean(int64_t dim) const {
    return ops::mean<T>(*this, dim);
}

template <typename T>
Tensor<T> Tensor<T>::max() const {
    return ops::max<T>(*this);
}

template <typename T>
Tensor<T> Tensor<T>::min() const {
    return ops::min<T>(*this);
}

template <typename T>
Tensor<T> Tensor<T>::softmax(int64_t dim) const {
    return ops::softmax<T>(*this, dim);
}

// ----- linalg operations -----

template <typename T>
Tensor<T> Tensor<T>::dot(const Tensor<T>& other) const {
    return ops::dot<T>(*this, other);
}

template <typename T>
Tensor<T> Tensor<T>::matmul(const Tensor<T>& other) const {
    return ops::matmul<T>(*this, other);
}

// ----- grad / backward -----

// Return the accumulated gradient tensor, or nullptr if backward has not
// been called yet or this tensor does not require grad.
template <typename T>
const Tensor<T>* Tensor<T>::grad() const {
    if (!grad_storage_ || !grad_storage_->tensor) return nullptr;
    return grad_storage_->tensor.get();
}

// Accumulate `g` into this leaf tensor's GradStorage.
// Called by the engine for each leaf input encountered during backward.
template <typename T>
void Tensor<T>::accumulate_grad(const Tensor<T>& g) {
    if (!grad_storage_) return;
    if (!grad_storage_->tensor) {
        grad_storage_->tensor = std::make_shared<Tensor<T>>(g.shape(), g.data(), false);
    } else {
        auto& dst = grad_storage_->tensor->data();
        const auto& src = g.data();
        for (size_t i = 0; i < dst.size(); ++i)
            dst[i] += src[i];
    }
}

// Reset the gradient to nullptr while keeping grad_storage_ alive, so aliases
// created in the next forward pass continue to share the same GradStorage.
template <typename T>
void Tensor<T>::zero_grad() const {
    if (grad_storage_) grad_storage_->tensor = nullptr;
}

// Kick off the backward pass from this tensor.
// Requires this tensor to be scalar (numel == 1) and to have been produced by
// an operation (grad_fn != nullptr). Gradients are accumulated into .grad()
// of every leaf tensor that requires_grad.
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
