#pragma once

#include <algorithm>   // std::copy
#include <cmath>       // std::sqrt
#include <cstdint>     // int64_t, uint64_t
#include <limits>      // std::numeric_limits
#include <memory>      // std::shared_ptr
#include <optional>    // std::optional, std::nullopt
#include <random>      // std::mt19937_64, distributions
#include <stdexcept>   // std::invalid_argument, std::overflow_error
#include <string>      // std::string
#include <type_traits> // std::is_same
#include <utility>     // std::move, std::pair
#include <vector>      // std::vector

#include "dtype.h"
#include "tensor_accessor.h"

namespace autograd {
template <typename T>
class Node;
} // namespace autograd

namespace tensor {

template <typename T>
class Tensor {
private:
    // shape of the tensor (dimensions)
    std::vector<int64_t> shape_;
    // strides of the tensor (how to access the elements in the tensor)
    std::vector<int64_t> strides_;
    // dtype of the tensor
    Dtype dtype_;
    // offset of the tensor in the storage (for views)
    int64_t offset_;
    // data pointer to the tensor (for owning tensors)
    std::shared_ptr<std::vector<T>> data_ptr_;
    // base pointer to the tensor (for views)
    std::shared_ptr<Tensor<T>> base_ptr_;
    // gradient tracking flag
    bool requires_grad_;
    // gradient function that produced this tensor
    std::shared_ptr<autograd::Node<T>> grad_fn_;

    // Gradient storage for leaf tensors (requires_grad=true, no grad_fn).
    //
    // Allocated at construction for leaves so alias() can copy the shared_ptr
    // and both the original tensor and every saved alias point to the same
    // GradStorage object.  accumulate_grad() writes into GradStorage::tensor;
    // zero_grad() resets it to nullptr.  grad() returns GradStorage::tensor.get().
    //
    // Null for operation results (non-leaves) and no-grad tensors.
    struct GradStorage {
        std::shared_ptr<Tensor<T>> tensor;  // null until first accumulate_grad()
    };
    mutable std::shared_ptr<GradStorage> grad_storage_;

    /**
     * Infers the runtime dtype tag from the compile-time storage type T.
     * Maps float/double/int32_t/int64_t onto the Dtype enum.
     *
     * @return The dtype corresponding to T.
     *
     * @throws std::invalid_argument if T is not float, double, int32_t, or int64_t.
     */
    static Dtype infer_dtype() {
        if (std::is_same<T, float>::value)
            return Dtype::Float32;
        else if (std::is_same<T, double>::value)
            return Dtype::Float64;
        else if (std::is_same<T, int32_t>::value)
            return Dtype::Int32;
        else if (std::is_same<T, int64_t>::value)
            return Dtype::Int64;
        else
            throw std::invalid_argument(
                "unsupported tensor element type; allowed types are float, double, int32_t, int64_t");
    }

    /**
     * Returns whether gradient tracking should be enabled for this storage type.
     * True only when `requires_grad` is true and neither `dtype` nor T is integer.
     *
     * @param requires_grad Requested gradient-tracking flag.
     * @param dtype Runtime dtype tag of the tensor.
     * @return True if the tensor should track gradients.
     */
    static bool grad_allowed(bool requires_grad, Dtype dtype) {
        const bool int_dtype = dtype == Dtype::Int32 || dtype == Dtype::Int64;
        const bool int_storage = std::is_same<T, int32_t>::value
                              || std::is_same<T, int64_t>::value;
        return requires_grad && !int_dtype && !int_storage;
    }

    /**
     * Checks that every dimension in `shape` is non-negative.
     * Returns the same vector on success so it can be used in initializer lists.
     *
     * @param shape Shape vector to validate.
     * @return `shape` unchanged.
     *
     * @throws std::invalid_argument if any dimension is negative.
     */
    static const std::vector<int64_t>& validate_shape(const std::vector<int64_t>& shape) {
        for (const int64_t dimension : shape)
            if (dimension < 0)
                throw std::invalid_argument("tensor shape must be non-negative");
        return shape;
    }

    /**
     * Multiplies two 64-bit integers with overflow detection.
     * Treats a zero operand as a zero product without checking overflow.
     *
     * @param lhs Left factor.
     * @param rhs Right factor.
     * @return `lhs * rhs`.
     *
     * @throws std::overflow_error if the product would exceed int64_t.
     */
    static int64_t safe_multiply(int64_t lhs, int64_t rhs) {
        if (lhs == 0 || rhs == 0)
            return 0;
        if (lhs > 0 && rhs > 0 && lhs > (std::numeric_limits<int64_t>::max() / rhs))
            throw std::overflow_error("tensor numel overflow");
        return lhs * rhs;
    }

    /**
     * Builds an owning tensor from already-allocated contiguous storage.
     * Private so callers cannot skip the size check against `shape`.
     *
     * @param shape Logical shape of the tensor.
     * @param storage Buffer moved into tensor-owned storage; size must equal numel(shape).
     * @param requires_grad Requested gradient-tracking flag (ignored for integer T).
     * @param grad_fn Backward node that produced this tensor, or nullptr for a leaf.
     *
     * @throws std::invalid_argument if any shape dimension is negative.
     * @throws std::invalid_argument if `storage.size()` does not match the shape.
     * @throws std::overflow_error if numel overflows int64_t.
     */
    Tensor(
        const std::vector<int64_t>& shape,
        std::vector<T>&& storage,
        bool requires_grad,
        std::shared_ptr<autograd::Node<T>> grad_fn
    ) :
        shape_(validate_shape(shape)),
        strides_(compute_contiguous_strides(shape_)),
        dtype_(infer_dtype()),
        offset_(0),
        data_ptr_(std::make_shared<std::vector<T>>(std::move(storage))),
        base_ptr_(nullptr),
        requires_grad_(grad_allowed(requires_grad, infer_dtype())),
        grad_fn_(std::move(grad_fn)),
        // Allocate GradStorage only for leaves (grad_fn_ == nullptr) that require grad.
        // Operation results have a grad_fn and must not accumulate grad here.
        grad_storage_(!grad_fn_ && grad_allowed(requires_grad, infer_dtype())
                      ? std::make_shared<GradStorage>() : nullptr) {
        const int64_t expected = compute_numel(shape_);
        if (static_cast<int64_t>(data_ptr_->size()) != expected)
            throw std::invalid_argument("storage size does not match tensor shape");
    }

    /**
     * Write `n` packed row-major values into this tensor's logical layout.
     * Contiguous destinations use a dense copy from `offset_`; otherwise values
     * are scattered through `strides_`.
     *
     * @param in Pointer to `n` source values in row-major logical order.
     * @param n Number of values; must equal `numel()`.
     */
    void write_packed(const T* in, int64_t n) {
        T* buf = data_ptr_->data();
        if (is_contiguous()) {
            std::copy(in, in + n, buf + offset_);
            return;
        }
        const int64_t r = rank();
        if (r == 0) {
            buf[offset_] = in[0];
            return;
        }
        std::vector<int64_t> idx(static_cast<size_t>(r), 0);
        for (int64_t i = 0; i < n; ++i) {
            int64_t off = offset_;
            for (int64_t d = 0; d < r; ++d)
                off += idx[static_cast<size_t>(d)] * strides_[static_cast<size_t>(d)];
            buf[off] = in[static_cast<size_t>(i)];
            for (int64_t d = r - 1; d >= 0; --d) {
                auto& v = idx[static_cast<size_t>(d)];
                if (++v < shape_[static_cast<size_t>(d)])
                    break;
                v = 0;
            }
        }
    }

public:
    // ----- helper methods -----

    /**
     * Computes the number of elements implied by `shape` (product of dimensions).
     * Validates non-negativity, then multiplies with overflow checks.
     *
     * @param shape Shape whose product is computed.
     * @return Product of the dimensions (1 for an empty shape / scalar).
     *
     * @throws std::invalid_argument if any dimension is negative.
     * @throws std::overflow_error if the product overflows int64_t.
     */
    static int64_t compute_numel(const std::vector<int64_t>& shape) {
        validate_shape(shape);
        int64_t total = 1;
        for (const int64_t dimension : shape)
            total = safe_multiply(total, dimension);
        return total;
    }

    /**
     * Computes row-major contiguous strides for `shape`.
     * Example: [2, 3, 4] -> [12, 4, 1].
     *
     * @param shape Shape to derive strides from.
     * @return Stride vector of the same rank as `shape`.
     *
     * @throws std::invalid_argument if any dimension is negative.
     * @throws std::overflow_error if a running product overflows int64_t.
     */
    static std::vector<int64_t> compute_contiguous_strides(const std::vector<int64_t>& shape) {
        validate_shape(shape);
        std::vector<int64_t> strides(shape.size(), 1);
        int64_t running = 1;
        for (int64_t i = static_cast<int64_t>(shape.size()) - 1; i >= 0; --i) {
            strides[static_cast<size_t>(i)] = running;
            running = safe_multiply(running, shape[static_cast<size_t>(i)]);
        }
        return strides;
    }

    /**
     * Maps a possibly negative dimension index into `[0, rank)`.
     * Negative indices count from the end (`-1` is the last axis).
     *
     * @param dim Dimension index to normalize.
     * @param rank Rank of the tensor (`ndim`).
     * @return Normalized index in `[0, rank)`.
     *
     * @throws std::out_of_range if `dim` is outside `[-rank, rank)`.
     */
    static int64_t normalize_dimension(int64_t dim, int64_t rank) {
        if (dim < 0)
            dim += rank;
        if (dim < 0 || dim >= rank)
            throw std::out_of_range("dimension index out of range");
        return dim;
    }

    // ----- tensor constructors -----

    /**
     * Constructs a contiguous tensor of `shape` with value-initialized elements.
     * Allocates storage of size `numel(shape)` and fills with T{}.
     *
     * @param shape Logical shape of the tensor.
     * @param requires_grad Whether the tensor should track gradients (default: true).
     *
     * @throws std::invalid_argument if any shape dimension is negative.
     * @throws std::overflow_error if numel overflows int64_t.
     */
    explicit Tensor(
        const std::vector<int64_t>& shape,
        bool requires_grad = true
    ) :
        shape_(validate_shape(shape)),
        strides_(compute_contiguous_strides(shape_)),
        dtype_(infer_dtype()),
        offset_(0),
        // T{} performs zero-initialization
        data_ptr_(std::make_shared<std::vector<T>>(static_cast<size_t>(compute_numel(shape_)), T{})),
        base_ptr_(nullptr),
        requires_grad_(grad_allowed(requires_grad, infer_dtype())),
        grad_fn_(nullptr),
        grad_storage_(grad_allowed(requires_grad, infer_dtype())
                  ? std::make_shared<GradStorage>() : nullptr) {}

    /**
     * Constructs a contiguous tensor of `shape` filled with `fill_value`.
     * Delegates to the zero-init constructor, then overwrites storage.
     *
     * @param shape Logical shape of the tensor.
     * @param fill_value Value written to every element.
     * @param requires_grad Whether the tensor should track gradients (default: true).
     *
     * @throws std::invalid_argument if any shape dimension is negative.
     * @throws std::overflow_error if numel overflows int64_t.
     */
    Tensor(
        const std::vector<int64_t>& shape,
        T fill_value,
        bool requires_grad = true
    ) : Tensor<T>(shape, requires_grad) {
        std::fill(data_ptr_->begin(), data_ptr_->end(), fill_value);
    }

    /**
     * Constructs a contiguous tensor of `shape` by copying `values`.
     * Dtype is always inferred from T.
     *
     * @param shape Logical shape of the tensor.
     * @param values Source values copied into new storage; size must equal numel(shape).
     * @param requires_grad Whether the tensor should track gradients (default: true).
     *
     * @throws std::invalid_argument if any shape dimension is negative.
     * @throws std::invalid_argument if `values.size()` does not match the shape.
     * @throws std::overflow_error if numel overflows int64_t.
     */
    Tensor(
        const std::vector<int64_t>& shape,
        const std::vector<T>& values,
        bool requires_grad = true
    ) :
        shape_(validate_shape(shape)),
        strides_(compute_contiguous_strides(shape_)),
        dtype_(infer_dtype()),
        offset_(0),
        data_ptr_(std::make_shared<std::vector<T>>(values)),
        base_ptr_(nullptr),
        requires_grad_(grad_allowed(requires_grad, infer_dtype())),
        grad_fn_(nullptr),
        grad_storage_(grad_allowed(requires_grad, infer_dtype())
                  ? std::make_shared<GradStorage>() : nullptr) {
        const int64_t expected = compute_numel(shape_);
        if (static_cast<int64_t>(values.size()) != expected)
            throw std::invalid_argument("input vector size does not match tensor shape");
    }

    /**
     * Constructs a contiguous tensor of `shape` by copying `n_elements` from a raw array.
     * Dtype is always inferred from T.
     *
     * @param shape Logical shape of the tensor.
     * @param values Pointer to the source array; may be null only when `n_elements == 0`.
     * @param n_elements Number of elements to copy; must equal numel(shape).
     * @param requires_grad Whether the tensor should track gradients (default: true).
     *
     * @throws std::invalid_argument if any shape dimension is negative.
     * @throws std::invalid_argument if `n_elements` is negative.
     * @throws std::invalid_argument if `n_elements` does not match the shape.
     * @throws std::invalid_argument if `values` is null and `n_elements > 0`.
     * @throws std::overflow_error if numel overflows int64_t.
     */
    Tensor(
        const std::vector<int64_t>& shape,
        const T* values,
        int64_t n_elements,
        bool requires_grad = true
    ) :
        shape_(validate_shape(shape)),
        strides_(compute_contiguous_strides(shape_)),
        dtype_(infer_dtype()),
        offset_(0),
        data_ptr_(nullptr),
        base_ptr_(nullptr),
        requires_grad_(grad_allowed(requires_grad, infer_dtype())),
        grad_fn_(nullptr),
        grad_storage_(nullptr) {
        if (n_elements < 0)
            throw std::invalid_argument("number of array elements must be non-negative");
        const int64_t expected = compute_numel(shape_);
        if (n_elements != expected)
            throw std::invalid_argument("number of array elements does not match tensor shape");
        if (values == nullptr && n_elements > 0)
            throw std::invalid_argument("array pointer must not be null when number of elements > 0");
        data_ptr_ = std::make_shared<std::vector<T>>(values, values + static_cast<size_t>(n_elements));
        grad_storage_ = grad_allowed(requires_grad_, infer_dtype())
                    ? std::make_shared<GradStorage>() : nullptr;
    }

    // ----- tensor factory methods -----

    /**
     * Builds a tensor from an operation's output buffer and optional backward node.
     * Moves `storage` into the private owning constructor.
     *
     * @param shape Logical shape of the result.
     * @param storage Buffer moved into tensor-owned storage; size must equal numel(shape).
     * @param requires_grad Whether the result should track gradients.
     * @param grad_fn Backward node that produced this tensor, or nullptr.
     * @return Owning tensor wrapping `storage`.
     *
     * @throws std::invalid_argument if any shape dimension is negative.
     * @throws std::invalid_argument if `storage.size()` does not match the shape.
     * @throws std::overflow_error if numel overflows int64_t.
     */
    static Tensor<T> from_operation_result(
        const std::vector<int64_t>& shape,
        std::vector<T>&& storage,
        bool requires_grad,
        std::shared_ptr<autograd::Node<T>> grad_fn
    ) {
        return Tensor<T>(shape, std::move(storage), requires_grad, std::move(grad_fn));
    }

    /**
     * Creates a lightweight alias that shares storage and GradStorage with `source`.
     * Copies the tensor then clears `grad_fn_` so the alias is not a graph node.
     * In-place mutation of the source is visible on the alias (no version counter).
     *
     * @param source Tensor to alias.
     * @return Tensor sharing `data_ptr_`, `base_ptr_`, and `grad_storage_` with `source`.
     */
    static Tensor<T> alias(const Tensor<T>& source) {
        Tensor<T> t(source);
        t.grad_fn_ = nullptr;
        return t;
    }

    /**
     * Builds a view that shares storage with `source` under new layout metadata.
     * Copies `source`, then overwrites shape/strides/offset and attaches `grad_fn`.
     * `grad_storage_` is cleared because operation-produced views are non-leaves.
     *
     * @param source Tensor whose storage is shared.
     * @param shape Logical shape of the view.
     * @param strides Per-dimension strides of the view (in elements).
     * @param offset Starting index into the shared buffer.
     * @param requires_grad Whether the view should track gradients.
     * @param grad_fn Backward node for this view op, or nullptr.
     * @return View tensor sharing storage with `source`.
     *
     * @throws std::invalid_argument if any shape dimension is negative.
     */
    static Tensor<T> from_view(
        const Tensor<T>& source,
        std::vector<int64_t> shape,
        std::vector<int64_t> strides,
        int64_t offset,
        bool requires_grad,
        std::shared_ptr<autograd::Node<T>> grad_fn
    ) {
        validate_shape(shape);
        Tensor<T> view(source);
        view.shape_ = std::move(shape);
        view.strides_ = std::move(strides);
        view.offset_ = offset;
        view.base_ptr_ = std::make_shared<Tensor<T>>(source);
        view.requires_grad_ = requires_grad;
        view.grad_fn_ = std::move(grad_fn);
        view.grad_storage_ = nullptr;
        return view;
    }

    // ----- copy / move constructors -----

    /**
     * Copy-constructs a tensor. Copies metadata and shared_ptr handles so storage is shared.
     *
     * @param other Tensor to copy.
     */
    Tensor(const Tensor<T>& other) = default;

    /**
     * Copy-assigns a tensor. Copies metadata and shared_ptr handles so storage is shared.
     *
     * @param other Tensor to copy.
     * @return `*this`.
     */
    auto operator=(const Tensor<T>& other) -> Tensor<T>& = default;

    /**
     * Move-constructs a tensor. Moves metadata and shared_ptr handles (handle semantics).
     *
     * @param other Tensor to move from.
     */
    Tensor(Tensor<T>&& other) noexcept = default;

    /**
     * Move-assigns a tensor. Moves metadata and shared_ptr handles (handle semantics).
     *
     * @param other Tensor to move from.
     * @return `*this`.
     */
    auto operator=(Tensor<T>&& other) noexcept -> Tensor<T>& = default;

    // ----- fill constructors -----

    /**
     * Constructs a contiguous tensor of `shape` filled with zeros.
     * Equivalent to `Tensor(shape, T{0}, requires_grad)`.
     *
     * @param shape Logical shape of the tensor.
     * @param requires_grad Whether the tensor should track gradients (default: true).
     * @return New tensor filled with T{0}.
     *
     * @throws std::invalid_argument if any shape dimension is negative.
     * @throws std::overflow_error if numel overflows int64_t.
     */
    static Tensor<T> zeros(
        const std::vector<int64_t>& shape,
        bool requires_grad = true
    ) {
        return Tensor<T>(shape, static_cast<T>(0), requires_grad);
    }

    /**
     * Constructs a contiguous tensor of `shape` filled with ones.
     * Equivalent to `Tensor(shape, T{1}, requires_grad)`.
     *
     * @param shape Logical shape of the tensor.
     * @param requires_grad Whether the tensor should track gradients (default: true).
     * @return New tensor filled with T{1}.
     *
     * @throws std::invalid_argument if any shape dimension is negative.
     * @throws std::overflow_error if numel overflows int64_t.
     */
    static Tensor<T> ones(
        const std::vector<int64_t>& shape,
        bool requires_grad = true
    ) {
        return Tensor<T>(shape, static_cast<T>(1), requires_grad);
    }

    /**
     * Constructs a contiguous tensor of `shape` filled with `fill_value`.
     * Equivalent to the fill-value constructor.
     *
     * @param shape Logical shape of the tensor.
     * @param fill_value Value written to every element.
     * @param requires_grad Whether the tensor should track gradients (default: true).
     * @return New tensor filled with `fill_value`.
     *
     * @throws std::invalid_argument if any shape dimension is negative.
     * @throws std::overflow_error if numel overflows int64_t.
     */
    static Tensor<T> full(
        const std::vector<int64_t>& shape,
        T fill_value,
        bool requires_grad = true
    ) {
        return Tensor<T>(shape, fill_value, requires_grad);
    }

    /**
     * Constructs a zero-filled tensor with the same shape as `other`.
     * Does not copy values or strides from `other`.
     *
     * @param other Tensor whose shape is copied.
     * @param requires_grad Whether the tensor should track gradients (default: true).
     * @return New tensor filled with T{0}.
     *
     * @throws std::invalid_argument if any shape dimension is negative.
     * @throws std::overflow_error if numel overflows int64_t.
     */
    static Tensor<T> zeros_like(
        const Tensor<T>& other,
        bool requires_grad = true
    ) {
        return Tensor<T>(other.shape_, static_cast<T>(0), requires_grad);
    }

    /**
     * Constructs a ones-filled tensor with the same shape as `other`.
     * Does not copy values or strides from `other`.
     *
     * @param other Tensor whose shape is copied.
     * @param requires_grad Whether the tensor should track gradients (default: true).
     * @return New tensor filled with T{1}.
     *
     * @throws std::invalid_argument if any shape dimension is negative.
     * @throws std::overflow_error if numel overflows int64_t.
     */
    static Tensor<T> ones_like(
        const Tensor<T>& other,
        bool requires_grad = true
    ) {
        return Tensor<T>(other.shape_, static_cast<T>(1), requires_grad);
    }

    /**
     * Constructs a filled tensor with the same shape as `other`.
     * Does not copy values or strides from `other`.
     *
     * @param other Tensor whose shape is copied.
     * @param fill_value Value written to every element.
     * @param requires_grad Whether the tensor should track gradients (default: true).
     * @return New tensor filled with `fill_value`.
     *
     * @throws std::invalid_argument if any shape dimension is negative.
     * @throws std::overflow_error if numel overflows int64_t.
     */
    static Tensor<T> full_like(
        const Tensor<T>& other,
        T fill_value,
        bool requires_grad = true
    ) {
        return Tensor<T>(other.shape_, fill_value, requires_grad);
    }

    /**
     * Constructs a 1-D tensor with values in the half-open range [start, end).
     * Count is `ceil((end - start) / step)` when the range is non-empty for `step`.
     *
     * @param start First value.
     * @param end Exclusive upper bound.
     * @param step Spacing between values (default: 1).
     * @param requires_grad Whether the tensor should track gradients (default: true).
     * @return 1-D tensor of values `start, start+step, ...` still in range.
     *
     * @throws std::invalid_argument if `step` is zero.
     * @throws std::invalid_argument if any shape dimension is negative (via the vector constructor).
     * @throws std::overflow_error if numel overflows int64_t.
     */
    static Tensor<T> arange(
        int64_t start,
        int64_t end,
        int64_t step = 1,
        bool requires_grad = true
    ) {
        if (step == 0)
            throw std::invalid_argument("arange: step must be non-zero");

        int64_t n = 0;
        if (step > 0 && end > start)
            n = (end - start + step - 1) / step;
        else if (step < 0 && end < start)
            n = (start - end - step - 1) / (-step);

        std::vector<T> storage(static_cast<size_t>(n));
        for (int64_t i = 0; i < n; ++i)
            storage[static_cast<size_t>(i)] = static_cast<T>(start + i * step);
        return Tensor<T>({n}, storage, requires_grad);
    }

    // ----- random initialization -----

    /**
     * Constructs a tensor of `shape` filled with integers uniformly in [low, high).
     * Always built with `requires_grad == false`.
     *
     * @param shape Logical shape of the tensor.
     * @param low Inclusive lower bound of the integer range.
     * @param high Exclusive upper bound of the integer range.
     * @param seed Optional RNG seed; if nullopt, `std::random_device` is used.
     * @return New tensor of samples in [low, high), without gradient tracking.
     *
     * @throws std::invalid_argument if `low >= high`.
     * @throws std::invalid_argument if any shape dimension is negative.
     * @throws std::overflow_error if numel overflows int64_t.
     */
    static Tensor<T> randint(
        const std::vector<int64_t>& shape,
        int64_t low,
        int64_t high,
        std::optional<uint64_t> seed = std::nullopt
    ) {
        if (low >= high)
            throw std::invalid_argument("randint: low must be strictly less than high");
        const int64_t n = compute_numel(shape);
        std::vector<T> storage(static_cast<size_t>(n));
        std::mt19937_64 rng{seed.has_value() ? *seed : std::random_device{}()};
        std::uniform_int_distribution<int64_t> dist(low, high - 1);
        for (auto& v : storage)
            v = static_cast<T>(dist(rng));
        return Tensor<T>(shape, storage, false);
    }

    /**
     * Constructs a tensor of `shape` filled with samples from N(0, 1).
     * Draws from a double-precision normal distribution and casts to T.
     *
     * @param shape Logical shape of the tensor.
     * @param requires_grad Whether the tensor should track gradients (default: true).
     * @param seed Optional RNG seed; if nullopt, `std::random_device` is used.
     * @return New tensor of standard-normal samples.
     *
     * @throws std::invalid_argument if any shape dimension is negative.
     * @throws std::overflow_error if numel overflows int64_t.
     */
    static Tensor<T> randn(
        const std::vector<int64_t>& shape,
        bool requires_grad = true,
        std::optional<uint64_t> seed = std::nullopt
    ) {
        const int64_t n = compute_numel(shape);
        std::vector<T> storage(static_cast<size_t>(n));
        std::mt19937_64 rng{seed.has_value() ? *seed : std::random_device{}()};
        std::normal_distribution<double> dist(0.0, 1.0);
        for (auto& v : storage)
            v = static_cast<T>(dist(rng));
        return Tensor<T>(shape, storage, requires_grad);
    }

    /**
     * Constructs a tensor of `shape` filled with samples from N(mean, stddev).
     * Draws from a double-precision normal distribution and casts to T.
     *
     * @param shape Logical shape of the tensor.
     * @param mean Mean of the normal distribution.
     * @param stddev Standard deviation of the normal distribution.
     * @param requires_grad Whether the tensor should track gradients (default: true).
     * @param seed Optional RNG seed; if nullopt, `std::random_device` is used.
     * @return New tensor of Gaussian samples.
     *
     * @throws std::invalid_argument if any shape dimension is negative.
     * @throws std::overflow_error if numel overflows int64_t.
     */
    static Tensor<T> random_gaussian(
        const std::vector<int64_t>& shape,
        T mean,
        T stddev,
        bool requires_grad = true,
        std::optional<uint64_t> seed = std::nullopt
    ) {
        const int64_t n = compute_numel(shape);
        std::vector<T> storage(static_cast<size_t>(n));
        std::mt19937_64 rng{seed.has_value() ? *seed : std::random_device{}()};
        std::normal_distribution<double> dist(
            static_cast<double>(mean),
            static_cast<double>(stddev)
        );
        for (auto& v : storage)
            v = static_cast<T>(dist(rng));
        return Tensor<T>(shape, storage, requires_grad);
    }

    /**
     * Constructs a randint tensor with the same shape as `other`.
     * Forwards to `randint(other.shape(), low, high, seed)`.
     *
     * @param other Tensor whose shape is copied.
     * @param low Inclusive lower bound of the integer range.
     * @param high Exclusive upper bound of the integer range.
     * @param seed Optional RNG seed; if nullopt, `std::random_device` is used.
     * @return New tensor of samples in [low, high), without gradient tracking.
     *
     * @throws std::invalid_argument if `low >= high`.
     * @throws std::invalid_argument if any shape dimension is negative.
     * @throws std::overflow_error if numel overflows int64_t.
     */
    static Tensor<T> randint_like(
        const Tensor<T>& other,
        int64_t low,
        int64_t high,
        std::optional<uint64_t> seed = std::nullopt
    ) {
        return randint(other.shape_, low, high, seed);
    }

    /**
     * Constructs a standard-normal tensor with the same shape as `other`.
     * Forwards to `randn(other.shape(), requires_grad, seed)`.
     *
     * @param other Tensor whose shape is copied.
     * @param requires_grad Whether the tensor should track gradients (default: true).
     * @param seed Optional RNG seed; if nullopt, `std::random_device` is used.
     * @return New tensor of standard-normal samples.
     *
     * @throws std::invalid_argument if any shape dimension is negative.
     * @throws std::overflow_error if numel overflows int64_t.
     */
    static Tensor<T> randn_like(
        const Tensor<T>& other,
        bool requires_grad = true,
        std::optional<uint64_t> seed = std::nullopt
    ) {
        return randn(other.shape_, requires_grad, seed);
    }

    /**
     * Constructs a Gaussian tensor with the same shape as `other`.
     * Forwards to `random_gaussian(other.shape(), mean, stddev, requires_grad, seed)`.
     *
     * @param other Tensor whose shape is copied.
     * @param mean Mean of the normal distribution.
     * @param stddev Standard deviation of the normal distribution.
     * @param requires_grad Whether the tensor should track gradients (default: true).
     * @param seed Optional RNG seed; if nullopt, `std::random_device` is used.
     * @return New tensor of Gaussian samples.
     *
     * @throws std::invalid_argument if any shape dimension is negative.
     * @throws std::overflow_error if numel overflows int64_t.
     */
    static Tensor<T> random_gaussian_like(
        const Tensor<T>& other,
        T mean,
        T stddev,
        bool requires_grad = true,
        std::optional<uint64_t> seed = std::nullopt
    ) {
        return random_gaussian(other.shape_, mean, stddev, requires_grad, seed);
    }

    // ----- destruction -----

    /**
     * Destroys the tensor. Releases shared_ptr handles; storage is freed when the last owner is gone.
     */
    ~Tensor() = default;

    // ----- init methods -----

    /**
     * Computes fan_in and fan_out for a weight tensor.
     * Rank-1 uses `{shape[0], shape[0]}`; rank >= 2 uses in/out from the first two dims
     * and folds remaining kernel dims into both fans.
     *
     * @param shape Shape of the weight tensor (rank >= 1 for the rank-1 branch).
     * @return Pair `{fan_in, fan_out}`.
     */
    static std::pair<int64_t, int64_t> compute_fans(const std::vector<int64_t>& shape) {
        if (shape.size() < 2)
            return {shape[0], shape[0]};
        int64_t fan_in  = shape[1];
        int64_t fan_out = shape[0];
        for (size_t i = 2; i < shape.size(); ++i) {
            fan_in  *= shape[i];
            fan_out *= shape[i];
        }
        return {fan_in, fan_out};
    }

    /**
     * Xavier / Glorot uniform initialization: U[-limit, limit]
     * where limit = gain * sqrt(6 / (fan_in + fan_out)).
     *
     * @param shape Logical shape of the tensor.
     * @param gain Scaling factor (default: 1.0).
     * @param requires_grad Whether the tensor should track gradients (default: true).
     * @param seed Optional RNG seed; if nullopt, `std::random_device` is used.
     * @return New tensor initialized with Xavier uniform samples.
     *
     * @throws std::invalid_argument if any shape dimension is negative.
     * @throws std::overflow_error if numel overflows int64_t.
     */
    static Tensor<T> xavier_uniform(
        const std::vector<int64_t>& shape,
        double gain = 1.0,
        bool requires_grad = true,
        std::optional<uint64_t> seed = std::nullopt
    ) {
        auto [fan_in, fan_out] = compute_fans(shape);
        const double limit = gain * std::sqrt(6.0 / static_cast<double>(fan_in + fan_out));
        const int64_t n = compute_numel(shape);
        std::vector<T> storage(static_cast<size_t>(n));
        std::mt19937_64 rng{seed.has_value() ? *seed : std::random_device{}()};
        std::uniform_real_distribution<double> dist(-limit, limit);
        for (auto& v : storage)
            v = static_cast<T>(dist(rng));
        return Tensor<T>(shape, storage, requires_grad);
    }

    /**
     * Xavier / Glorot normal initialization: N(0, std)
     * where std = gain * sqrt(2 / (fan_in + fan_out)).
     *
     * @param shape Logical shape of the tensor.
     * @param gain Scaling factor (default: 1.0).
     * @param requires_grad Whether the tensor should track gradients (default: true).
     * @param seed Optional RNG seed; if nullopt, `std::random_device` is used.
     * @return New tensor initialized with Xavier normal samples.
     *
     * @throws std::invalid_argument if any shape dimension is negative.
     * @throws std::overflow_error if numel overflows int64_t.
     */
    static Tensor<T> xavier_normal(
        const std::vector<int64_t>& shape,
        double gain = 1.0,
        bool requires_grad = true,
        std::optional<uint64_t> seed = std::nullopt
    ) {
        auto [fan_in, fan_out] = compute_fans(shape);
        const double std = gain * std::sqrt(2.0 / static_cast<double>(fan_in + fan_out));
        const int64_t n = compute_numel(shape);
        std::vector<T> storage(static_cast<size_t>(n));
        std::mt19937_64 rng{seed.has_value() ? *seed : std::random_device{}()};
        std::normal_distribution<double> dist(0.0, std);
        for (auto& v : storage)
            v = static_cast<T>(dist(rng));
        return Tensor<T>(shape, storage, requires_grad);
    }

    /**
     * Kaiming / He uniform initialization: U[-bound, bound]
     * where bound = sqrt(3) * gain / sqrt(fan) and gain = sqrt(2 / (1 + negative_slope^2)).
     *
     * @param shape Logical shape of the tensor.
     * @param negative_slope Negative slope of the nonlinearity (0.0 for ReLU; default: 0.0).
     * @param fan_mode `"fan_in"` (default) or `"fan_out"`.
     * @param requires_grad Whether the tensor should track gradients (default: true).
     * @param seed Optional RNG seed; if nullopt, `std::random_device` is used.
     * @return New tensor initialized with Kaiming uniform samples.
     *
     * @throws std::invalid_argument if `fan_mode` is neither `"fan_in"` nor `"fan_out"`.
     * @throws std::invalid_argument if any shape dimension is negative.
     * @throws std::overflow_error if numel overflows int64_t.
     */
    static Tensor<T> kaiming_uniform(
        const std::vector<int64_t>& shape,
        double negative_slope = 0.0,
        const std::string& fan_mode = "fan_in",
        bool requires_grad = true,
        std::optional<uint64_t> seed = std::nullopt
    ) {
        auto [fan_in, fan_out] = compute_fans(shape);
        int64_t fan;
        if (fan_mode == "fan_in")
            fan = fan_in;
        else if (fan_mode == "fan_out")
            fan = fan_out;
        else
            throw std::invalid_argument("kaiming_uniform: fan_mode must be \"fan_in\" or \"fan_out\"");
        const double gain  = std::sqrt(2.0 / (1.0 + negative_slope * negative_slope));
        const double bound = std::sqrt(3.0) * gain / std::sqrt(static_cast<double>(fan));
        const int64_t n = compute_numel(shape);
        std::vector<T> storage(static_cast<size_t>(n));
        std::mt19937_64 rng{seed.has_value() ? *seed : std::random_device{}()};
        std::uniform_real_distribution<double> dist(-bound, bound);
        for (auto& v : storage)
            v = static_cast<T>(dist(rng));
        return Tensor<T>(shape, storage, requires_grad);
    }

    /**
     * Kaiming / He normal initialization: N(0, std)
     * where std = gain / sqrt(fan) and gain = sqrt(2 / (1 + negative_slope^2)).
     *
     * @param shape Logical shape of the tensor.
     * @param negative_slope Negative slope of the nonlinearity (0.0 for ReLU; default: 0.0).
     * @param fan_mode `"fan_in"` (default) or `"fan_out"`.
     * @param requires_grad Whether the tensor should track gradients (default: true).
     * @param seed Optional RNG seed; if nullopt, `std::random_device` is used.
     * @return New tensor initialized with Kaiming normal samples.
     *
     * @throws std::invalid_argument if `fan_mode` is neither `"fan_in"` nor `"fan_out"`.
     * @throws std::invalid_argument if any shape dimension is negative.
     * @throws std::overflow_error if numel overflows int64_t.
     */
    static Tensor<T> kaiming_normal(
        const std::vector<int64_t>& shape,
        double negative_slope = 0.0,
        const std::string& fan_mode = "fan_in",
        bool requires_grad = true,
        std::optional<uint64_t> seed = std::nullopt
    ) {
        auto [fan_in, fan_out] = compute_fans(shape);
        int64_t fan;
        if (fan_mode == "fan_in")
            fan = fan_in;
        else if (fan_mode == "fan_out")
            fan = fan_out;
        else
            throw std::invalid_argument("kaiming_normal: fan_mode must be \"fan_in\" or \"fan_out\"");
        const double gain = std::sqrt(2.0 / (1.0 + negative_slope * negative_slope));
        const double std  = gain / std::sqrt(static_cast<double>(fan));
        const int64_t n = compute_numel(shape);
        std::vector<T> storage(static_cast<size_t>(n));
        std::mt19937_64 rng{seed.has_value() ? *seed : std::random_device{}()};
        std::normal_distribution<double> dist(0.0, std);
        for (auto& v : storage)
            v = static_cast<T>(dist(rng));
        return Tensor<T>(shape, storage, requires_grad);
    }

    /**
     * Uniform initialization: U[low, high) over `shape`.
     * Draws from a double-precision uniform distribution and casts to T.
     *
     * @param shape Logical shape of the tensor.
     * @param low Inclusive lower bound.
     * @param high Exclusive upper bound.
     * @param requires_grad Whether the tensor should track gradients (default: true).
     * @param seed Optional RNG seed; if nullopt, `std::random_device` is used.
     * @return New tensor of uniform samples.
     *
     * @throws std::invalid_argument if `low >= high`.
     * @throws std::invalid_argument if any shape dimension is negative.
     * @throws std::overflow_error if numel overflows int64_t.
     */
    static Tensor<T> uniform(
        const std::vector<int64_t>& shape,
        T low,
        T high,
        bool requires_grad = true,
        std::optional<uint64_t> seed = std::nullopt
    ) {
        if (!(low < high))
            throw std::invalid_argument("uniform: low must be < high");
        const int64_t n = compute_numel(shape);
        std::vector<T> storage(static_cast<size_t>(n));
        std::mt19937_64 rng{seed.has_value() ? *seed : std::random_device{}()};
        std::uniform_real_distribution<double> dist(
            static_cast<double>(low), static_cast<double>(high));
        for (auto& v : storage)
            v = static_cast<T>(dist(rng));
        return Tensor<T>(shape, storage, requires_grad);
    }

    // ----- accessors -----

    /**
     * Returns the number of dimensions (size of `shape_`). Same as `ndim()`.
     *
     * @return Rank as a signed integer.
     */
    int64_t rank() const {
        return static_cast<int64_t>(shape_.size());
    }

    /**
     * Returns the number of logical elements (product of `shape_`).
     * Independent of strides, offset, and backing-buffer size.
     *
     * @return Element count (0 if any dimension is 0).
     *
     * @throws std::overflow_error if the product overflows int64_t.
     */
    int64_t numel() const {
        return compute_numel(shape_);
    }

    /**
     * Returns the logical shape of the tensor.
     *
     * @return Const reference to the shape vector.
     */
    const std::vector<int64_t>& shape() const {
        return shape_;
    }

    /**
     * Returns the number of dimensions. Alias of `rank()`.
     *
     * @return Rank as `shape_.size()` (unsigned converted to int64_t by the return type).
     */
    int64_t ndim() const {
        return shape_.size();
    }

    /**
     * Returns per-dimension steps in the backing buffer, in elements (not bytes).
     * Logical index (i0, i1, ...) maps to `offset_ + i0*s0 + i1*s1 + ...`.
     *
     * @return Const reference to the stride vector.
     */
    const std::vector<int64_t>& strides() const {
        return strides_;
    }

    /**
     * Returns the starting index into the backing buffer (0 for owners).
     *
     * @return Element offset (not a byte offset).
     */
    int64_t offset() const {
        return offset_;
    }

    /**
     * Returns the runtime dtype tag inferred from T at construction.
     *
     * @return One of Float32, Float64, Int32, Int64.
     */
    Dtype dtype() const {
        return dtype_;
    }

    /**
     * Returns whether this tensor participates in autograd.
     * Always false for integer storage types.
     *
     * @return True if gradients should be tracked.
     */
    bool requires_grad() const {
        return requires_grad_;
    }

    /**
     * Returns whether this tensor is a view of another tensor (`base_ptr_ != nullptr`).
     * Views share `data_ptr_` with the base; they do not own a separate buffer.
     *
     * @return True for views created by shape ops.
     */
    bool is_view() const {
        return base_ptr_ != nullptr;
    }

    /**
     * Returns whether strides match row-major contiguous strides for `shape_`.
     * Does not require `offset_ == 0`; a contiguous slice may still have a non-zero offset.
     *
     * @return True if `strides_ == compute_contiguous_strides(shape_)`.
     *
     * @throws std::overflow_error if computing contiguous strides overflows int64_t.
     */
    bool is_contiguous() const {
        return strides_ == compute_contiguous_strides(shape_);
    }

    /**
     * Returns the backward node that produced this tensor, or nullptr for a leaf.
     *
     * @return Shared pointer to the autograd node (may be empty).
     */
    const std::shared_ptr<autograd::Node<T>>& grad_fn() const {
        return grad_fn_;
    }

    /**
     * Returns the entire backing storage, not just this tensor's logical elements.
     * For a view the vector may be larger than `numel()`; use `offset()` and `strides()`.
     *
     * @return Const reference to the shared storage vector.
     */
    const std::vector<T>& data() const {
        return *data_ptr_;
    }

    /**
     * Returns mutable backing storage. Same caveats as the const overload.
     * In-place writes are not version-checked.
     *
     * @return Mutable reference to the shared storage vector.
     */
    std::vector<T>& data() {
        return *data_ptr_;
    }

    /**
     * Copies logical elements into a packed row-major vector.
     * Walks this tensor's shape in C-order; contiguous tensors copy a dense slice
     * from `offset_`, views are gathered through `strides_`.
     *
     * @return Vector of length `numel()` in row-major logical order.
     */
    std::vector<T> to_vector() const {
        const int64_t n = numel();
        std::vector<T> out(static_cast<size_t>(n));
        if (n == 0)
            return out;
        const T* buf = data_ptr_->data();
        if (is_contiguous()) {
            std::copy(buf + offset_, buf + offset_ + n, out.begin());
            return out;
        }
        const int64_t r = rank();
        if (r == 0) {
            out[0] = buf[offset_];
            return out;
        }
        std::vector<int64_t> idx(static_cast<size_t>(r), 0);
        for (int64_t i = 0; i < n; ++i) {
            int64_t off = offset_;
            for (int64_t d = 0; d < r; ++d)
                off += idx[static_cast<size_t>(d)] * strides_[static_cast<size_t>(d)];
            out[static_cast<size_t>(i)] = buf[off];
            for (int64_t d = r - 1; d >= 0; --d) {
                auto& v = idx[static_cast<size_t>(d)];
                if (++v < shape_[static_cast<size_t>(d)])
                    break;
                v = 0;
            }
        }
        return out;
    }

    /**
     * Copy logical values from `src` into this tensor, keeping this object's identity.
     * Shape must match. Storage, `requires_grad`, and `grad_fn` are unchanged so
     * optimizer pointers and leaf GradStorage stay valid. Overlapping views of the
     * same buffer go through a temporary packed copy.
     *
     * @param src Source tensor; logical shape must equal `shape_`.
     *
     * @throws std::invalid_argument if `src.shape()` differs from `shape_`.
     */
    void copy_(const Tensor<T>& src) {
        if (shape_ != src.shape_)
            throw std::invalid_argument("copy_: shape mismatch");
        if (this == &src)
            return;
        const int64_t n = numel();
        if (n == 0)
            return;
        if (src.is_contiguous() && data_ptr_ != src.data_ptr_) {
            write_packed(src.data_ptr_->data() + src.offset_, n);
            return;
        }
        const std::vector<T> packed = src.to_vector();
        write_packed(packed.data(), n);
    }

    /**
     * Indexes along the leading dimension via chained `operator[]`.
     * Builds a `TensorAccessor` at `offset_` and consumes `idx` on the first axis.
     *
     * @param idx Index along the leading remaining dimension.
     * @return Accessor for the next dimension, or a scalar proxy at rank 0.
     *
     * @throws std::out_of_range if there are no remaining dimensions.
     * @throws std::out_of_range if `idx` is outside the leading dimension.
     */
    TensorAccessor<T> operator[](int64_t idx) {
        return TensorAccessor<T>(
            data_ptr_->data() + offset_,
            shape_.data(),
            strides_.data(),
            rank()
        )[idx];
    }

    /**
     * Const overload of chained `operator[]`. Assignment through the returned accessor is disabled.
     *
     * @param idx Index along the leading remaining dimension.
     * @return Read-only accessor for the next dimension.
     *
     * @throws std::out_of_range if there are no remaining dimensions.
     * @throws std::out_of_range if `idx` is outside the leading dimension.
     */
    TensorAccessor<const T> operator[](int64_t idx) const {
        return TensorAccessor<const T>(
            data_ptr_->data() + offset_,
            shape_.data(),
            strides_.data(),
            rank()
        )[idx];
    }

    // ----- casting methods -----

    /**
     * Casts every element to scalar type U and returns a new packed tensor.
     * Packs non-contiguous sources first. Float→float keeps `requires_grad` and
     * attaches CastBackward when grad is enabled; integer results never require grad.
     *
     * @return New `Tensor<U>` with the same shape.
     */
    template <typename U>
    Tensor<U> to() const;

    /**
     * Casts this tensor to Float32. Wrapper for `to<float>()`.
     *
     * @return New Float32 tensor (tracks grad if this tensor does and grad is enabled).
     */
    Tensor<float> float32() const;

    /**
     * Casts this tensor to Float64. Wrapper for `to<double>()`.
     *
     * @return New Float64 tensor (tracks grad if this tensor does and grad is enabled).
     */
    Tensor<double> float64() const;

    /**
     * Casts this tensor to Int32. Wrapper for `to<int32_t>()`; result never requires grad.
     *
     * @return New Int32 tensor (non-grad leaf).
     */
    Tensor<int32_t> int32() const;

    /**
     * Casts this tensor to Int64. Wrapper for `to<int64_t>()`; result never requires grad.
     *
     * @return New Int64 tensor (non-grad leaf).
     */
    Tensor<int64_t> int64() const;

    // ----- shape operations -----

    /**
     * Reinterprets storage as `shape`. Zero-copy if already contiguous; otherwise packs first.
     * Delegates to `ops::reshape`.
     *
     * @param shape Target shape; product of dims must equal `numel()`.
     * @return View with contiguous strides for `shape`.
     *
     * @throws std::invalid_argument if the tensor is a scalar.
     * @throws std::invalid_argument if `numel` would change.
     * @throws std::invalid_argument if any dimension of `shape` is negative.
     * @throws std::overflow_error if computing numel of `shape` overflows int64_t.
     */
    Tensor<T> reshape(const std::vector<int64_t>& shape) const;

    /**
     * Swaps two dimensions by exchanging shape entries and strides (zero-copy).
     * Delegates to `ops::transpose`.
     *
     * @param dim0 First dimension (negative indices allowed).
     * @param dim1 Second dimension (negative indices allowed).
     * @return View with the two axes exchanged.
     *
     * @throws std::invalid_argument if the tensor is a scalar.
     * @throws std::out_of_range if a dimension is out of bounds.
     */
    Tensor<T> transpose(int64_t dim0, int64_t dim1) const;

    /**
     * Swaps the two axes of a rank-2 tensor. Equivalent to `transpose(0, 1)`.
     *
     * @return View with shape `[n, m]` if this tensor is `[m, n]`.
     *
     * @throws std::invalid_argument if rank is not 2.
     */
    Tensor<T> transpose() const;

    /**
     * Expands this tensor to `target_shape` by setting stride 0 on broadcast axes.
     * Zero-copy: elements are not repeated in memory. Delegates to `ops::broadcast_to`.
     *
     * @param target_shape Compatible shape (NumPy right-alignment rules).
     * @return View with stride 0 on expanded axes.
     *
     * @throws std::invalid_argument if target rank is less than source rank.
     * @throws std::invalid_argument if a dimension cannot broadcast.
     * @throws std::invalid_argument if any dimension of `target_shape` is negative.
     */
    Tensor<T> broadcast_to(const std::vector<int64_t>& target_shape) const;

    /**
     * Shares storage under a new contiguous shape. Never copies; throws if not contiguous.
     * Delegates to `ops::view`, which then calls `ops::reshape`.
     *
     * @param shape Target shape; product of dims must equal `numel()`.
     * @return View with contiguous strides for `shape`.
     *
     * @throws std::invalid_argument if the tensor is not contiguous.
     * @throws std::invalid_argument if the tensor is a scalar.
     * @throws std::invalid_argument if `numel` would change.
     * @throws std::invalid_argument if any dimension of `shape` is negative.
     */
    Tensor<T> view(const std::vector<int64_t>& shape) const;

    /**
     * Collapses dimensions `[start_dim, end_dim]` (inclusive) into one.
     * Packs first if the tensor is not contiguous. Delegates to `ops::flatten`.
     *
     * @param start_dim First collapsed axis (default 0; negative allowed).
     * @param end_dim Last collapsed axis (default -1; negative allowed).
     * @return View with those axes merged.
     *
     * @throws std::invalid_argument if the tensor is a scalar.
     * @throws std::invalid_argument if `start_dim > end_dim` after normalization.
     * @throws std::out_of_range if a dimension is out of bounds.
     */
    Tensor<T> flatten(int64_t start_dim = 0, int64_t end_dim = -1) const;

    /**
     * Removes every size-1 dimension. Remaining strides are kept, so this is valid on views.
     * Delegates to `ops::squeeze`.
     *
     * @return View with all size-1 axes dropped (scalar if every dim was 1).
     */
    Tensor<T> squeeze() const;

    /**
     * Removes the size-1 dimension at `dim`. No-op (still a view) if that dimension is not size 1.
     * Delegates to `ops::squeeze`.
     *
     * @param dim Axis to drop (negative indices allowed).
     * @return View with that axis removed, or an unchanged view.
     *
     * @throws std::out_of_range if `dim` is out of bounds.
     */
    Tensor<T> squeeze(int64_t dim) const;

    /**
     * Inserts a size-1 dimension at `dim`. Valid range is `[-rank-1, rank]`.
     * Delegates to `ops::unsqueeze`.
     *
     * @param dim Insertion index (negative indices allowed).
     * @return View with a new size-1 axis.
     *
     * @throws std::out_of_range if `dim` is outside `[-rank-1, rank]`.
     */
    Tensor<T> unsqueeze(int64_t dim) const;

    /**
     * Returns a tensor whose logical order is packed into a dense row-major buffer.
     * Already contiguous with `offset() == 0` returns `*this`; otherwise copies.
     * Delegates to `ops::contiguous`.
     *
     * @return Contiguous tensor with the same shape and logical values.
     */
    Tensor<T> contiguous() const;

    /**
     * Returns a view of a slice along `dim`: indices `[start, start+length)`.
     * Delegates to `ops::narrow`.
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
    Tensor<T> narrow(int64_t dim, int64_t start, int64_t length) const;

    // ----- math operations -----

    /**
     * Element-wise addition. Inputs must have the same shape (no kernel broadcasting).
     * Packs non-contiguous inputs first. Delegates to `ops::add`.
     *
     * @param other Right-hand operand.
     * @return New tensor `*this + other`.
     *
     * @throws std::invalid_argument on shape mismatch.
     */
    Tensor<T> add(const Tensor<T>& other) const;

    /**
     * Element-wise negation. Allocates a new buffer. Delegates to `ops::neg`.
     *
     * @return New tensor `-(*this)`.
     */
    Tensor<T> neg() const;

    /**
     * Element-wise subtraction. Same-shape inputs; new buffer. Delegates to `ops::subtract`.
     *
     * @param other Right-hand operand.
     * @return New tensor `*this - other`.
     *
     * @throws std::invalid_argument on shape mismatch.
     */
    Tensor<T> subtract(const Tensor<T>& other) const;

    /**
     * Element-wise multiplication. Same-shape inputs; new buffer. Delegates to `ops::multiply`.
     *
     * @param other Right-hand operand.
     * @return New tensor `*this * other`.
     *
     * @throws std::invalid_argument on shape mismatch.
     */
    Tensor<T> multiply(const Tensor<T>& other) const;

    /**
     * Element-wise division. Same-shape inputs; new buffer. Delegates to `ops::divide`.
     *
     * @param other Right-hand operand.
     * @return New tensor `*this / other`.
     *
     * @throws std::invalid_argument on shape mismatch.
     * @throws std::runtime_error on division by zero.
     */
    Tensor<T> divide(const Tensor<T>& other) const;

    /**
     * Element-wise power with a scalar exponent. New buffer. Delegates to `ops::power`.
     *
     * @param exponent Scalar exponent.
     * @return New tensor `(*this) ** exponent`.
     */
    Tensor<T> power(const T exponent) const;

    /**
     * Element-wise absolute value. New buffer. Delegates to `ops::abs`.
     *
     * @return New tensor `| *this |`.
     */
    Tensor<T> abs() const;

    /**
     * Element-wise exponential. New buffer. Delegates to `ops::exp`.
     *
     * @return New tensor `exp(*this)`.
     */
    Tensor<T> exp() const;

    /**
     * Element-wise natural logarithm. New buffer. Delegates to `ops::log`.
     *
     * @return New tensor `log(*this)`.
     *
     * @throws std::runtime_error if any element is non-positive.
     */
    Tensor<T> log() const;

    /**
     * Element-wise square root. New buffer. Delegates to `ops::sqrt`.
     *
     * @return New tensor `sqrt(*this)`.
     */
    Tensor<T> sqrt() const;

    /**
     * Element-wise sine. New buffer. Delegates to `ops::sin`.
     *
     * @return New tensor `sin(*this)`.
     */
    Tensor<T> sin() const;

    /**
     * Element-wise cosine. New buffer. Delegates to `ops::cos`.
     *
     * @return New tensor `cos(*this)`.
     */
    Tensor<T> cos() const;

    /**
     * Element-wise tangent. New buffer. Delegates to `ops::tan`.
     *
     * @return New tensor `tan(*this)`.
     */
    Tensor<T> tan() const;

    /**
     * Element-wise hyperbolic sine. New buffer. Delegates to `ops::sinh`.
     *
     * @return New tensor `sinh(*this)`.
     */
    Tensor<T> sinh() const;

    /**
     * Element-wise hyperbolic cosine. New buffer. Delegates to `ops::cosh`.
     *
     * @return New tensor `cosh(*this)`.
     */
    Tensor<T> cosh() const;

    /**
     * Element-wise hyperbolic tangent. New buffer. Delegates to `ops::tanh`.
     *
     * @return New tensor `tanh(*this)`.
     */
    Tensor<T> tanh() const;

    /**
     * Element-wise sigmoid, `1 / (1 + exp(-x))`. New buffer. Delegates to `ops::sigmoid`.
     *
     * @return New tensor in `(0, 1)`.
     */
    Tensor<T> sigmoid() const;

    /**
     * Element-wise ReLU, `max(0, x)`. New buffer. Delegates to `ops::relu`.
     *
     * @return New tensor with negatives zeroed.
     */
    Tensor<T> relu() const;

    /**
     * Element-wise SiLU (swish), `x * sigmoid(x)`. New buffer. Delegates to `ops::silu`.
     *
     * @return New tensor.
     */
    Tensor<T> silu() const;

    /**
     * Element-wise GELU (tanh approximation). New buffer. Delegates to `ops::gelu`.
     *
     * @return New tensor.
     */
    Tensor<T> gelu() const;

    // ----- reduction operations -----

    /**
     * Sums every element into a scalar tensor of shape `{}`. Delegates to `ops::sum`.
     *
     * @return Scalar tensor.
     *
     * @throws std::invalid_argument if the tensor is empty.
     */
    Tensor<T> sum() const;

    /**
     * Sums along one axis; that axis is removed from the output shape. Delegates to `ops::sum`.
     *
     * @param dim Axis to reduce (negative indices allowed).
     * @return Tensor with `dim` dropped.
     *
     * @throws std::invalid_argument if the tensor is a scalar.
     * @throws std::out_of_range if `dim` is out of bounds.
     */
    Tensor<T> sum(int64_t dim) const;

    /**
     * Mean of every element as a scalar tensor of shape `{}`. Delegates to `ops::mean`.
     *
     * @return Scalar tensor.
     *
     * @throws std::invalid_argument if the tensor is empty.
     */
    Tensor<T> mean() const;

    /**
     * Mean along one axis; that axis is removed from the output shape. Delegates to `ops::mean`.
     *
     * @param dim Axis to reduce (negative indices allowed).
     * @return Tensor with `dim` dropped.
     *
     * @throws std::invalid_argument if the tensor is a scalar.
     * @throws std::invalid_argument if the reduced dimension has size 0.
     * @throws std::out_of_range if `dim` is out of bounds.
     */
    Tensor<T> mean(int64_t dim) const;

    /**
     * Maximum element as a scalar tensor of shape `{}`. Ties keep the first index.
     * Delegates to `ops::max`.
     *
     * @return Scalar tensor.
     *
     * @throws std::invalid_argument if the tensor is empty.
     */
    Tensor<T> max() const;

    /**
     * Maximum along one axis; that axis is removed from the output shape.
     * Ties keep the first index (for backward). Delegates to `ops::max`.
     *
     * @param dim Axis to reduce (negative indices allowed).
     * @return Tensor with `dim` dropped.
     *
     * @throws std::invalid_argument if the tensor is a scalar.
     * @throws std::invalid_argument if the reduced dimension has size 0.
     * @throws std::out_of_range if `dim` is out of bounds.
     */
    Tensor<T> max(int64_t dim) const;

    /**
     * Minimum element as a scalar tensor of shape `{}`. Ties keep the first index.
     * Delegates to `ops::min`.
     *
     * @return Scalar tensor.
     *
     * @throws std::invalid_argument if the tensor is empty.
     */
    Tensor<T> min() const;

    /**
     * Minimum along one axis; that axis is removed from the output shape.
     * Ties keep the first index (for backward). Delegates to `ops::min`.
     *
     * @param dim Axis to reduce (negative indices allowed).
     * @return Tensor with `dim` dropped.
     *
     * @throws std::invalid_argument if the tensor is a scalar.
     * @throws std::invalid_argument if the reduced dimension has size 0.
     * @throws std::out_of_range if `dim` is out of bounds.
     */
    Tensor<T> min(int64_t dim) const;

    /**
     * Softmax along `dim` (max-subtraction for stability). Output shape matches input.
     * Delegates to `ops::softmax`.
     *
     * @param dim Axis to normalize (negative indices allowed).
     * @return Tensor of the same shape; each slice along `dim` sums to 1.
     *
     * @throws std::invalid_argument if the tensor is a scalar.
     * @throws std::invalid_argument if the reduced dimension has size 0.
     * @throws std::out_of_range if `dim` is out of bounds.
     */
    Tensor<T> softmax(int64_t dim) const;

    // ----- linalg operations -----

    /**
     * Inner product of two 1-D tensors of equal length. Result shape is `{}`.
     * Packs non-contiguous inputs first. Delegates to `ops::dot`.
     *
     * @param other Other vector.
     * @return Scalar tensor.
     *
     * @throws std::invalid_argument if either input is not 1-D or lengths differ.
     */
    Tensor<T> dot(const Tensor<T>& other) const;

    /**
     * Matrix product on the last two dimensions. Leading dims are batch axes and are broadcast.
     * Reads through strides. Delegates to `ops::matmul`.
     *
     * @param other Right-hand tensor of shape `[..., K, N]` if `*this` is `[..., M, K]`.
     * @return New contiguous tensor of shape `[broadcast(...), M, N]`.
     *
     * @throws std::invalid_argument if either rank is below 2.
     * @throws std::invalid_argument if inner dims mismatch.
     * @throws std::invalid_argument if batch dims are not broadcast-compatible.
     */
    Tensor<T> matmul(const Tensor<T>& other) const;

    // ----- grad / backward -----

    /**
     * Returns whether this tensor was not produced by an op (`grad_fn_ == nullptr`).
     * User-created tensors are leaves; differentiable `to()` outputs are not.
     *
     * @return True for leaves.
     */
    bool is_leaf() const {
        return grad_fn_ == nullptr;
    }

    /**
     * Returns the accumulated gradient of a leaf, or nullptr if backward has not run.
     * Also nullptr if this tensor does not require grad (`grad_storage_` is null).
     *
     * @return Pointer into `GradStorage`, or nullptr.
     */
    const Tensor<T>* grad() const;

    /**
     * Adds `grad` into this leaf's gradient buffer. Engine-internal.
     * Packs `grad` then copies or accumulates element-wise. No-op if `grad_storage_` is null.
     *
     * @param grad Incoming gradient; should match this tensor's logical size.
     */
    void accumulate_grad(const Tensor<T>& grad);

    /**
     * Drops the gradient tensor but keeps `GradStorage` so later aliases still share it.
     * Call before each backward pass.
     */
    void zero_grad() const;

    /**
     * Runs reverse-mode autodiff from this scalar output.
     * Seeds the graph with 1 and accumulates into `.grad()` of reachable requiring-grad leaves.
     *
     * @return `*this` (unchanged).
     *
     * @throws std::invalid_argument if `numel() != 1`.
     * @throws std::invalid_argument if `grad_fn_` is null (leaf).
     */
    Tensor<T> backward() const;
};

} // namespace tensor
