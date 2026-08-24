#pragma once

/*
Tensor class
args:
- shape_ (dimensions)
- strides_ (interpretation of a contiguous memory block)
- dtype_ (one of Float32, Float64, Int32, Int64)
- offset_ (starting point in the contiguous memory block)
- data_ptr_ (pointer to the data buffer)
- base_ptr_ (pointer to the base tensor, for views)
- requires_grad_ (whether the tensor requires gradients)
- grad_fn_ (pointer to the autograd node that produced this tensor)

methods:
- accessors:
    - rank()
    - numel()
    - shape()
    - ndim()
    - strides()
    - offset()
    - dtype()
    - requires_grad()
    - is_view()
    - is_contiguous()
    - grad_fn()
    - data()
- casting methods:
    - to()
    - float32()
    - float64()
    - int32()
    - int64()
- fill constructors:
    - zeros()
    - ones()
    - full()
    - zeros_like()
    - ones_like()
    - full_like()
    - randint()
    - randn()
    - random_gaussian() (mean, stddev)
    - randint_like()
    - randn_like()
    - random_gaussian_like()
- shape operations:
    - reshape()
    - transpose()
    - broadcast_to()
    - view()
    - flatten()
    - squeeze()
    - unsqueeze()
    - contiguous()
- elementwise operations:
    - add()
    - neg()
    - subtract()
    - multiply()
    - divide()
    - power()
    - abs()
    - exp()
    - log()
    - sin()
    - cos()
    - tan()
    - sinh()
    - cosh()
    - tanh()
    - sigmoid()
    - relu()
    - silu()
    - gelu()
- reduction operations:
    - sum()
    - sum(dim)
    - mean()
    - mean(dim)
    - max()
    - min()
    - softmax(dim)
- backward:
    - backward()
*/



// This header intentionally focuses on one phase only:
// "Tensor construction and destruction".
//
// The class below contains:
// - a minimal memory model (owning tensor + optional base pointer for future views),
// - creation paths requested by the project roadmap,
// - explicit comments about ownership and lifetime semantics.

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

// enum class Dtype {
//     Float32,
//     Float64,
//     Int32,
//     Int64
// };

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
     * Infers the dtype of a tensor from the compile-time type T.
     *
     * @return The dtype of the tensor.
     * @throws std::invalid_argument if the type is not supported.
     */
    static Dtype infer_dtype() 
    {
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
     * Check if gradient tracking is allowed for a given dtype (integer dtypes cannot track gradients).
     * Checks both the explicitly requested dtype and the compile-time storage type T.
     *
     * @param[in] requires_grad Whether gradient tracking is required.
     * @param[in] dtype The dtype of the tensor.
     * @param[in] T The type of the tensor.
     * @return True if gradient tracking is allowed, false otherwise.
     */
    static bool grad_allowed(bool requires_grad, Dtype dtype) 
    {
        const bool int_dtype = dtype == Dtype::Int32 || dtype == Dtype::Int64;
        const bool int_storage = std::is_same<T, int32_t>::value
                              || std::is_same<T, int64_t>::value;
        return requires_grad && !int_dtype && !int_storage;
    }

    /**
     * Validates that all dimensions in a shape are non-negative.
     *
     * @param[in] shape The shape to validate.
     * @return The validated shape.
     * @throws std::invalid_argument if any dimension is negative.
     */
    static const std::vector<int64_t>& validate_shape(const std::vector<int64_t>& shape) 
    {
        for (const int64_t dimension : shape) 
        {
            if (dimension < 0)
                throw std::invalid_argument("tensor shape must be non-negative");
        }
        return shape;
    }

    /**
     * Performs a multiplication of two 64-bit integers with overflow detection.
     *
     * @param[in] lhs The left operand.
     * @param[in] rhs The right operand.
     * @return The product of the two operands
     * @throws std::overflow_error if the product overflows.
     */
    static int64_t safe_multiply(int64_t lhs, int64_t rhs) 
    {
        if (lhs == 0 || rhs == 0)
            return 0;
        if (lhs > 0 && rhs > 0 && lhs > (std::numeric_limits<int64_t>::max() / rhs))
            throw std::overflow_error("tensor numel overflow");
        return lhs * rhs;
    }

    // Internal constructor for owning tensors built from already prepared storage.
    // This constructor is intentionally private so users cannot bypass size checks.
    Tensor(
        const std::vector<int64_t>& shape,
        std::vector<T>&& storage,
        bool requires_grad,
        std::shared_ptr<autograd::Node<T>> grad_fn
    ):
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
                      ? std::make_shared<GradStorage>() : nullptr)
    {
        const int64_t expected = compute_numel(shape_);
        if (static_cast<int64_t>(data_ptr_->size()) != expected) {
            throw std::invalid_argument("storage size does not match tensor shape");
        }
    }

public:
    // ----- helper methods -----

    /**
     * Computes the number of elements in a tensor given its shape.
     *
     * @param[in] shape The shape of the tensor.
     * @return The number of elements in the tensor.
     * @throws std::invalid_argument if the shape is negative.
     */
    static int64_t compute_numel(const std::vector<int64_t>& shape) {
        validate_shape(shape);
        int64_t total = 1;
        for (const int64_t dimension : shape) 
        {
            total = safe_multiply(total, dimension);
        }
        return total;
    }

    /**
     * Computes the contiguous strides for a given shape.
     * Example: [2, 3, 4] -> [12, 4, 1].
     *
     * @param[in] shape The shape of the tensor.
     * @return The contiguous strides for the tensor.
     */
    static std::vector<int64_t> compute_contiguous_strides(
        const std::vector<int64_t>& shape
    ) {
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
     * Normalizes a possibly-negative dimension index and validates bounds.
     * Example for rank=4: -1 -> 3, -4 -> 0.
     *
     * @param[in] dim The dimension index to normalize.
     * @param[in] rank The rank of the tensor.
     * @return The normalized dimension index.
     * @throws std::out_of_range if the dimension index is out of range.
     */
    static int64_t normalize_dimension(int64_t dim, int64_t rank) 
    {
        if (dim < 0)
            dim += rank;
        if (dim < 0 || dim >= rank)
            throw std::out_of_range("dimension index out of range");
        return dim;
    }

    // ----- tensor constructors -----

    /**
     * Constructs a new tensor with the given shape.
     * Allocates contiguous storage and value-initializes elements ("zero-initialized").
     *
     * @param[in] shape The shape of the tensor.
     * @param[in] requires_grad Whether the tensor requires gradients (default: true)
     * @return A new tensor with the given shape.
     */
    explicit Tensor(
        const std::vector<int64_t>& shape,
        bool requires_grad = true
    ):
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
                  ? std::make_shared<GradStorage>() : nullptr)
    {}

    /**
     * Constructs a new tensor with the given shape and fill value.
     * Used for "zeros/ones/full-like" style creation.
     *
     * @param[in] shape The shape of the tensor.
     * @param[in] fill_value The value to fill the tensor with.
     * @param[in] requires_grad Whether the tensor requires gradients (default: true)
     * @return A new tensor with the given shape and fill value.
     */
    Tensor(
        const std::vector<int64_t>& shape,
        T fill_value,
        bool requires_grad = true
    ): Tensor<T>(shape, requires_grad)
    {
        // zero-initialize the tensor and fill it with the fill value
        std::fill(data_ptr_->begin(), data_ptr_->end(), fill_value);
    }

    /**
     * Constructs a new tensor with the given shape and values from std::vector.
     * The dtype is always inferred from the template type T.
     * To construct with a different scalar type, instantiate Tensor<U> directly,
     * or call .to<U>() after construction.
     *
     * @param[in] shape The shape of the tensor.
     * @param[in] values The values to fill the tensor with.
     *                   The vector is copied into new tensor-owned storage.
     * @param[in] requires_grad Whether the tensor requires gradients (default: true)
     * @return A new tensor with the given shape and values.
     * @throws std::invalid_argument if the vector size does not match specified shape.
     */
    Tensor(
        const std::vector<int64_t>& shape,
        const std::vector<T>& values,
        bool requires_grad = true
    ):
        shape_(validate_shape(shape)),
        strides_(compute_contiguous_strides(shape_)),
        dtype_(infer_dtype()),
        offset_(0),
        // copy the vector into new tensor-owned storage
        data_ptr_(std::make_shared<std::vector<T>>(values)),
        base_ptr_(nullptr),
        requires_grad_(grad_allowed(requires_grad, infer_dtype())),
        grad_fn_(nullptr),
        grad_storage_(grad_allowed(requires_grad, infer_dtype())
                  ? std::make_shared<GradStorage>() : nullptr)
    {
        const int64_t expected = compute_numel(shape_);
        if (static_cast<int64_t>(values.size()) != expected)
            throw std::invalid_argument("input vector size does not match tensor shape");
    }

    /**
     * Constructs a new tensor with the given shape and values from raw array with specified size.
     * The dtype is always inferred from the template type T.
     * To construct with a different scalar type, instantiate Tensor<U> directly,
     * or call .to<U>() after construction.
     *
     * @param[in] shape The shape of the tensor.
     * @param[in] values The pointer to the raw array with value to copy into the tensor.
     * @param[in] count The number of elements in the array.
     * @param[in] requires_grad Whether the tensor requires gradients (default: true)
     * @return A new tensor with the given shape and values.
     * @throws std::invalid_argument if the vector size does not match specified shape.
     */
    Tensor(
        const std::vector<int64_t>& shape,
        const T* values,
        int64_t count,
        bool requires_grad = true
    ):
        shape_(validate_shape(shape)),
        strides_(compute_contiguous_strides(shape_)),
        dtype_(infer_dtype()),
        offset_(0),
        data_ptr_(nullptr),
        base_ptr_(nullptr),
        requires_grad_(grad_allowed(requires_grad, infer_dtype())),
        grad_fn_(nullptr),
        grad_storage_(nullptr)   // set after validation below
    {   
        // validate the array pointer and count
        if (count < 0)
            throw std::invalid_argument("array element count must be non-negative");
        const int64_t expected = compute_numel(shape_);
        if (count != expected)
            throw std::invalid_argument("array element count does not match tensor shape");
        if (values == nullptr && count > 0)
            throw std::invalid_argument("array pointer must not be null when count > 0");
        // copy the array into new tensor-owned storage
        data_ptr_ = std::make_shared<std::vector<T>>(values, values + static_cast<size_t>(count));
        grad_storage_ = grad_allowed(requires_grad_, infer_dtype())
                    ? std::make_shared<GradStorage>() : nullptr;
    }

    // ----- tensor factory methods -----

    /**
     * Constructs a new tensor with the given shape and values from std::vector (using std::move).
     * If the tensor is result of some operation, it gets assigned the given gradient function.
     *
     * @param[in] shape The shape of the tensor.
     * @param[in] storage The storage to move into new tensor-owned storage.
     * @param[in] requires_grad Whether the tensor requires gradients (default: true)
     * @param[in] grad_fn The gradient function that produced this tensor (default: nullptr).
     * @return A new tensor with the given shape and storage.
     */
    static Tensor<T> from_operation_result(
        const std::vector<int64_t>& shape,
        std::vector<T>&& storage,
        bool requires_grad,
        std::shared_ptr<autograd::Node<T>> grad_fn
    ) {
        return Tensor<T>(shape, std::move(storage), requires_grad, std::move(grad_fn));
    }

    // Creates a lightweight alias that shares the same underlying storage as
    // `source` without copying any data.
    //
    // Both data_ptr_ and grad_storage_ are copied (shared_ptr copy — no allocation).
    // Because leaf tensors pre-allocate their GradStorage at construction time,
    // the alias and the original share the same GradStorage object.
    // When the engine calls accumulate_grad() on the alias, GradStorage::tensor
    // is set and immediately visible on the original through tensor.grad().
    //
    // requires_grad_ is kept so the engine can distinguish leaf-grad inputs
    // (needs accumulation) from no-grad inputs (nullptr edge, skip).
    // grad_fn_ is stripped because an alias is not itself a graph operation.
    //
    // Caution: in-place mutation of the source between forward and backward
    // will be observed by the alias (no version counter guard here).

    // Caution: if the source tensor is mutated in-place after the forward pass
    // but before backward(), the backward computation will observe the mutated
    // values. This mirrors PyTorch's behaviour (which detects such mutations via
    // version counters and raises an error; we omit the guard here).

    /**
     * Creates identical view of the tensor.
     * By using copy constructor, we create a new reference (by increasing ref count) 
     * to data_ptr_, base_ptr_, grad_fn_ and accumulate_grad_fn_. 
     * All other metadata are deep copied. We then disconnect it from computation graph
     * (decreasing ref count by setting it to nullptr).   
     *
     * @param[in,out] source The tensor to create alias of.
     * @return A new tensor that is an alias of the tensor.
     */
    static Tensor<T> alias(const Tensor<T>& source) 
    {
        Tensor<T> t(source);    // copies all shared_ptrs including grad_storage_
        t.grad_fn_ = nullptr;   // strip: alias is not a graph operation result
        return t;
    }

    // Factory for view tensors: shares storage with source, uses caller-supplied
    // shape/strides/offset so shape-manipulating ops (transpose, etc.) need no
    // direct access to private fields.
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
        view.grad_fn_       = std::move(grad_fn);
        view.grad_storage_      = nullptr; // operation-produced views are non-leaf
        return view;
    }

    // ----- copy / move constructors -----

    /**
     * Copy constructor and copy assignment: default because it copies metadata and shared_ptr handles.
     * Copies share the same storage by default (handle semantics)
     */
    Tensor(const Tensor<T>& other) = default;
    auto operator=(const Tensor<T>& other) -> Tensor<T>& = default;
 
     /**
      * Move constructor and move assignment: default because it moves metadata and shared_ptr handles.
      * Moves share the same storage by default (handle semantics)
      */
    Tensor(Tensor<T>&& other) noexcept = default;
    auto operator=(Tensor<T>&& other) noexcept -> Tensor<T>& = default;


    // ----- fill constructors -----

    /**
     * Constructs a new tensor with the given shape and filled with zeros.
     *
     * @param[in] shape The shape of the tensor.
     * @param[in] requires_grad Whether the tensor requires gradients (default: true)
     * @return A new tensor with the given shape and filled with zeros.
     */
    static Tensor<T> zeros(const std::vector<int64_t>& shape, bool requires_grad = true) 
    {
        return Tensor<T>(shape, static_cast<T>(0), requires_grad);
    }

    /**
     * Constructs a new tensor with the given shape and filled with ones.
     *
     * @param[in] shape The shape of the tensor.
     * @param[in] requires_grad Whether the tensor requires gradients (default: true)
     * @return A new tensor with the given shape and filled with ones.
     */
    static Tensor<T> ones(const std::vector<int64_t>& shape, bool requires_grad = true) 
    {
        return Tensor<T>(shape, static_cast<T>(1), requires_grad);
    }

    /**
     * Constructs a new tensor with the given shape and filled with a specific value.
     *
     * @param[in] shape The shape of the tensor.
     * @param[in] requires_grad Whether the tensor requires gradients (default: true)
     * @return A new tensor with the given shape and filled with the specified value.
     */
    static Tensor<T> full(const std::vector<int64_t>& shape, T fill_value, bool requires_grad = true) 
    {
        return Tensor<T>(shape, fill_value, requires_grad);
    }

    /**
     * Constructs a new tensor with the same shape as the other tensor and filled with zeros.
     *
     * @param[in] other The other tensor to copy the shape from.
     * @param[in] requires_grad Whether the tensor requires gradients (default: true)
     * @return A new tensor with the same shape as the other tensor and filled with zeros.
     */
    static Tensor<T> zeros_like(const Tensor<T>& other, bool requires_grad = true) 
    {
        return Tensor<T>(other.shape_, static_cast<T>(0), requires_grad);
    }

    /**
     * Constructs a new tensor with the same shape as the other tensor and filled with ones.
     *
     * @param[in] other The other tensor to copy the shape from.
     * @param[in] requires_grad Whether the tensor requires gradients (default: true)
     * @return A new tensor with the same shape as the other tensor and filled with ones.
     */
    static Tensor<T> ones_like(const Tensor<T>& other, bool requires_grad = true) 
    {
        return Tensor<T>(other.shape_, static_cast<T>(1), requires_grad);
    }

    /**
     * Constructs a new tensor with the same shape as the other tensor and filled with a specific value.
     *
     * @param[in] other The other tensor to copy the shape from.
     * @param[in] requires_grad Whether the tensor requires gradients (default: true)
     * @return A new tensor with the same shape as the other tensor and filled with the specified value.
     */
    static Tensor<T> full_like(const Tensor<T>& other, T fill_value, bool requires_grad = true) 
    {
        return Tensor<T>(other.shape_, fill_value, requires_grad);
    }

    // ----- random initialization -----

    /**
     * Constructs a new tensor with the given shape and filled with integers in range [low, high).
     *
     * @param[in] shape The shape of the tensor.
     * @param[in] low The lower bound of the uniform distribution.
     * @param[in] high The upper bound of the uniform distribution.
     * @param[in] requires_grad Whether the tensor requires gradients (default: false)
     * @param[in] seed The optional seed for the random number generator (default: std::nullopt)
     * @return A new tensor with the given shape and filled with integers in range [low, high).
     * @throws std::invalid_argument if low >= high.
     */
    static Tensor<T> randint(
        const std::vector<int64_t>& shape,
        int64_t low,
        int64_t high,
        bool requires_grad = false,
        std::optional<uint64_t> seed = std::nullopt
    ) {
        if (low >= high)
            throw std::invalid_argument("randint: low must be strictly less than high");
        const int64_t n = compute_numel(shape);
        std::vector<T> storage(static_cast<size_t>(n));
        // initialize the random number generator with the optional seed
        std::mt19937_64 rng{seed.has_value() ? *seed : std::random_device{}()};
        std::uniform_int_distribution<int64_t> dist(low, high - 1);
        // fill the tensor with random integers
        for (auto& v : storage)
            v = static_cast<T>(dist(rng));
        return Tensor<T>::from_operation_result(shape, std::move(storage), requires_grad, nullptr);
    }

    /**
     * Constructs a new tensor with the given shape and filled with samples from N(0, 1).
     *
     * @param[in] shape The shape of the tensor.
     * @param[in] requires_grad Whether the tensor requires gradients (default: true)
     * @param[in] seed The optional seed for the random number generator (default: std::nullopt)
     * @return A new tensor with the given shape and filled with samples from N(0, 1).
     */
    static Tensor<T> randn(
        const std::vector<int64_t>& shape,
        bool requires_grad = true,
        std::optional<uint64_t> seed = std::nullopt
    ) {
        const int64_t n = compute_numel(shape);
        std::vector<T> storage(static_cast<size_t>(n));
        // initialize the random number generator with the optional seed
        std::mt19937_64 rng{seed.has_value() ? *seed : std::random_device{}()};
        std::normal_distribution<double> dist(0.0, 1.0);
        // fill the tensor with random samples from N(0, 1)
        for (auto& v : storage)
            v = static_cast<T>(dist(rng));
        return Tensor<T>::from_operation_result(shape, std::move(storage), requires_grad, nullptr);
    }

    /**
     * Constructs a new tensor with the given shape and filled with samples from N(mean, stddev).
     *
     * @param[in] shape The shape of the tensor.
     * @param[in] mean The mean of the normal distribution.
     * @param[in] stddev The standard deviation of the normal distribution.
     * @param[in] requires_grad Whether the tensor requires gradients (default: true)
     * @param[in] seed The optional seed for the random number generator (default: std::nullopt)
     * @return A new tensor with the given shape and filled with samples from N(mean, stddev).
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
        // initialize the random number generator with the optional seed
        std::mt19937_64 rng{seed.has_value() ? *seed : std::random_device{}()};
        std::normal_distribution<double> dist(
            static_cast<double>(mean),
            static_cast<double>(stddev)
        );
        // fill the tensor with random samples from N(mean, stddev)
        for (auto& v : storage)
            v = static_cast<T>(dist(rng));
        return Tensor<T>::from_operation_result(shape, std::move(storage), requires_grad, nullptr);
    }

    /**
     * Constructs a new tensor with the same shape as the other tensor and filled with integers in range [low, high).
     *
     * @param[in] other The other tensor to copy the shape from.
     * @param[in] low The lower bound of the uniform distribution.
     * @param[in] high The upper bound of the uniform distribution.
     * @param[in] requires_grad Whether the tensor requires gradients (default: true)
     * @param[in] seed The optional seed for the random number generator (default: std::nullopt)
     * @return A new tensor with the same shape as the other tensor and filled with integers in range [low, high).
     */
    static Tensor<T> randint_like(
        const Tensor<T>& other,
        int64_t low,
        int64_t high,
        bool requires_grad = false,
        std::optional<uint64_t> seed = std::nullopt
    ) {
        return randint(other.shape_, low, high, requires_grad, seed);
    }

    /**
     * Constructs a new tensor with the same shape as the other tensor and filled with samples from N(0, 1).
     *
     * @param[in] other The other tensor to copy the shape from.
     * @param[in] requires_grad Whether the tensor requires gradients (default: true)
     * @param[in] seed The optional seed for the random number generator (default: std::nullopt)
     * @return A new tensor with the same shape as the other tensor and filled with samples from N(0, 1).
     */
    static Tensor<T> randn_like(
        const Tensor<T>& other,
        bool requires_grad = true,
        std::optional<uint64_t> seed = std::nullopt
    ) {
        return randn(other.shape_, requires_grad, seed);
    }

    /**
     * Constructs a new tensor with the same shape as the other tensor and filled with samples from N(mean, stddev).
     *
     * @param[in] other The other tensor to copy the shape from.
     * @param[in] mean The mean of the normal distribution.
     * @param[in] stddev The standard deviation of the normal distribution.
     * @param[in] requires_grad Whether the tensor requires gradients (default: true)
     * @param[in] seed The optional seed for the random number generator (default: std::nullopt)
     * @return A new tensor with the same shape as the other tensor and filled with samples from N(mean, stddev).
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
     * Destructor: default because all resources are RAII-managed (std::vector and std::shared_ptr).
     * Destroying a tensor decrements references and frees data when the last owner is gone.
     */
    ~Tensor() = default;

    // ----- init methods -----

    /**
     * Computes fan_in and fan_out for a weight tensor.
     *
     * For a 2-D tensor of shape [out, in]: fan_in = in, fan_out = out.
     * For higher-rank tensors (e.g. conv weights [out, in, *k]):
     *   the kernel dimensions are folded into both fan values.
     *
     * @param[in] shape The shape of the weight tensor (rank >= 1).
     * @return A pair {fan_in, fan_out}.
     */
    static std::pair<int64_t, int64_t> compute_fans(const std::vector<int64_t>& shape) 
    {
        if (shape.size() < 2) {
            return {shape[0], shape[0]};
        }
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
     * @param[in] shape The shape of the tensor.
     * @param[in] gain  Scaling factor (default: 1.0).
     * @param[in] requires_grad Whether the tensor requires gradients (default: true).
     * @param[in] seed  Optional RNG seed.
     * @return A new tensor initialized with Xavier uniform values.
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
        return Tensor<T>::from_operation_result(shape, std::move(storage), requires_grad, nullptr);
    }

    /**
     * Xavier / Glorot normal initialization: N(0, std)
     * where std = gain * sqrt(2 / (fan_in + fan_out)).
     *
     * @param[in] shape The shape of the tensor.
     * @param[in] gain  Scaling factor (default: 1.0).
     * @param[in] requires_grad Whether the tensor requires gradients (default: true).
     * @param[in] seed  Optional RNG seed.
     * @return A new tensor initialized with Xavier normal values.
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
        return Tensor<T>::from_operation_result(shape, std::move(storage), requires_grad, nullptr);
    }

    /**
     * Kaiming / He uniform initialization: U[-bound, bound]
     * where bound = sqrt(3) * gain / sqrt(fan)
     * and   gain  = sqrt(2 / (1 + negative_slope^2)).
     *
     * @param[in] shape          The shape of the tensor.
     * @param[in] negative_slope Negative slope of the activation (0.0 for ReLU, default: 0.0).
     * @param[in] fan_mode       Which fan to use: "fan_in" (default) or "fan_out".
     * @param[in] requires_grad  Whether the tensor requires gradients (default: true).
     * @param[in] seed           Optional RNG seed.
     * @return A new tensor initialized with Kaiming uniform values.
     * @throws std::invalid_argument if fan_mode is neither "fan_in" nor "fan_out".
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
        if (fan_mode == "fan_in")       fan = fan_in;
        else if (fan_mode == "fan_out") fan = fan_out;
        else throw std::invalid_argument("kaiming_uniform: fan_mode must be \"fan_in\" or \"fan_out\"");
        const double gain  = std::sqrt(2.0 / (1.0 + negative_slope * negative_slope));
        const double bound = std::sqrt(3.0) * gain / std::sqrt(static_cast<double>(fan));
        const int64_t n = compute_numel(shape);
        std::vector<T> storage(static_cast<size_t>(n));
        std::mt19937_64 rng{seed.has_value() ? *seed : std::random_device{}()};
        std::uniform_real_distribution<double> dist(-bound, bound);
        for (auto& v : storage)
            v = static_cast<T>(dist(rng));
        return Tensor<T>::from_operation_result(shape, std::move(storage), requires_grad, nullptr);
    }

    /**
     * Kaiming / He normal initialization: N(0, std)
     * where std  = gain / sqrt(fan)
     * and   gain = sqrt(2 / (1 + negative_slope^2)).
     *
     * @param[in] shape          The shape of the tensor.
     * @param[in] negative_slope Negative slope of the activation (0.0 for ReLU, default: 0.0).
     * @param[in] fan_mode       Which fan to use: "fan_in" (default) or "fan_out".
     * @param[in] requires_grad  Whether the tensor requires gradients (default: true).
     * @param[in] seed           Optional RNG seed.
     * @return A new tensor initialized with Kaiming normal values.
     * @throws std::invalid_argument if fan_mode is neither "fan_in" nor "fan_out".
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
        if (fan_mode == "fan_in")       fan = fan_in;
        else if (fan_mode == "fan_out") fan = fan_out;
        else throw std::invalid_argument("kaiming_normal: fan_mode must be \"fan_in\" or \"fan_out\"");
        const double gain = std::sqrt(2.0 / (1.0 + negative_slope * negative_slope));
        const double std  = gain / std::sqrt(static_cast<double>(fan));
        const int64_t n = compute_numel(shape);
        std::vector<T> storage(static_cast<size_t>(n));
        std::mt19937_64 rng{seed.has_value() ? *seed : std::random_device{}()};
        std::normal_distribution<double> dist(0.0, std);
        for (auto& v : storage)
            v = static_cast<T>(dist(rng));
        return Tensor<T>::from_operation_result(shape, std::move(storage), requires_grad, nullptr);
    }

    /**
     * Uniform initialization: U[low, high).
     *
     * @param[in] shape The shape of the tensor.
     * @param[in] low   Inclusive lower bound.
     * @param[in] high  Exclusive upper bound.
     * @param[in] requires_grad Whether the tensor requires gradients (default: true).
     * @param[in] seed  Optional RNG seed.
     * @return A new tensor filled with uniform samples.
     * @throws std::invalid_argument if low >= high.
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
        return Tensor<T>::from_operation_result(shape, std::move(storage), requires_grad, nullptr);
    }


    // ----- accessors -----

    /**
     * Number of dimensions (size of `shape_`). Same as `ndim()`.
     *
     * @return Rank as a signed integer.
     */
    int64_t rank() const {
        return static_cast<int64_t>(shape_.size());
    }

    /**
     * Number of logical elements, i.e. the product of `shape_`.
     * Independent of strides, offset, and backing-buffer size.
     *
     * @return Element count (0 for an empty dimension).
     */
    int64_t numel() const {
        return compute_numel(shape_);
    }

    /**
     * Logical shape of the tensor.
     *
     * @return Reference to the shape vector (do not mutate).
     */
    const std::vector<int64_t>& shape() const {
        return shape_;
    }

    /**
     * Number of dimensions. Alias of `rank()`.
     *
     * @return Rank as a signed integer.
     */
    int64_t ndim() const {
        return shape_.size();
    }

    /**
     * Per-dimension step in the backing buffer, in elements (not bytes).
     * Logical index (i0, i1, ...) maps to `offset_ + i0*s0 + i1*s1 + ...`.
     *
     * @return Reference to the stride vector (do not mutate).
     */
    const std::vector<int64_t>& strides() const {
        return strides_;
    }

    /**
     * Starting index into the backing buffer for this tensor (0 for owners).
     *
     * @return Element offset (not a byte offset).
     */
    int64_t offset() const {
        return offset_;
    }

    /**
     * Runtime dtype tag inferred from the template parameter T.
     *
     * @return One of Float32, Float64, Int32, Int64.
     */
    Dtype dtype() const {
        return dtype_;
    }

    /**
     * Whether this tensor participates in autograd.
     * Always false for integer storage types.
     *
     * @return True if gradients should be tracked.
     */
    bool requires_grad() const {
        return requires_grad_;
    }

    /**
     * Whether this tensor is a view of another tensor (`base_ptr_ != nullptr`).
     * Views share `data_ptr_` with the base; they do not own a separate buffer.
     *
     * @return True for views created by shape ops.
     */
    bool is_view() const {
        return base_ptr_ != nullptr;
    }

    /**
     * Whether strides match row-major contiguous strides for `shape_`.
     * Does not require `offset_ == 0`; a contiguous slice can still have a
     * non-zero offset.
     *
     * @return True if a dense `data()[i]` loop would match logical order
     *         only when `offset_ == 0` as well.
     */
    bool is_contiguous() const {
        return strides_ == compute_contiguous_strides(shape_);
    }

    /**
     * Backward node that produced this tensor, or nullptr for a leaf.
     *
     * @return Shared pointer to the autograd node (may be empty).
     */
    const std::shared_ptr<autograd::Node<T>>& grad_fn() const {
        return grad_fn_;
    }

    /**
     * Entire backing storage, not just this tensor's logical elements.
     * For a view the vector may be larger than `numel()`; use `offset()` and
     * `strides()` (or `operator[]`) to index logically. Dense kernels that
     * read `data()[i]` for `i in 0..numel()` assume a contiguous tensor with
     * `offset() == 0` — call `contiguous()` first if needed.
     *
     * @return Const reference to the shared storage vector.
     */
    const std::vector<T>& data() const {
        return *data_ptr_;
    }

    /**
     * Mutable backing storage. Same caveats as the const overload: this is
     * the full buffer, and in-place writes are not version-checked.
     *
     * @return Mutable reference to the shared storage vector.
     */
    std::vector<T>& data() {
        return *data_ptr_;
    }

    /**
     * Multi-dimensional element access via chained `operator[]`.
     * Usage: `tensor[i][j][k]`. Each call consumes one dimension using
     * `offset_` and `strides_`, so views (transpose, broadcast) index correctly.
     *
     * @param[in] idx Index along the leading remaining dimension.
     * @return Accessor for the next dimension, or a scalar proxy at rank 0.
     * @throws std::out_of_range if `idx` is out of bounds.
     */
    TensorAccessor<T> operator[](int64_t idx) 
    {
        return TensorAccessor<T>(
            data_ptr_->data() + offset_,
            shape_.data(),
            strides_.data(),
            rank()
        )[idx];
    }

    /**
     * Const overload of chained `operator[]`. Assignment through the
     * returned accessor is disabled.
     *
     * @param[in] idx Index along the leading remaining dimension.
     * @return Read-only accessor for the next dimension.
     * @throws std::out_of_range if `idx` is out of bounds.
     */
    TensorAccessor<const T> operator[](int64_t idx) const 
    {
        return TensorAccessor<const T>(
            data_ptr_->data() + offset_,
            shape_.data(),
            strides_.data(),
            rank()
        )[idx];
    }

    // ----- casting methods -----

    /**
     * Cast every element to scalar type U and return a new packed tensor.
     * Non-contiguous sources are packed first (logical order). Float→float
     * keeps `requires_grad` and attaches `CastBackward` (backward recasts
     * the gradient to the input dtype). Integer results never require grad.
     *
     * @return New `Tensor<U>` with the same shape.
     */
    template <typename U>
    Tensor<U> to() const;

    /**
     * Convenience wrapper for `to<float>()`.
     *
     * @return New Float32 tensor (tracks grad if this tensor does).
     */
    Tensor<float> float32() const;

    /**
     * Convenience wrapper for `to<double>()`.
     *
     * @return New Float64 tensor (tracks grad if this tensor does).
     */
    Tensor<double> float64() const;

    /**
     * Convenience wrapper for `to<int32_t>()`. Integer results never require grad.
     *
     * @return New Int32 tensor (non-grad leaf).
     */
    Tensor<int32_t> int32() const;

    /**
     * Convenience wrapper for `to<int64_t>()`. Integer results never require grad.
     *
     * @return New Int64 tensor (non-grad leaf).
     */
    Tensor<int64_t> int64() const;

    // ----- shape operations -----

    /**
     * Reinterpret the same storage as `shape` (zero-copy view).
     * `numel` must match and the tensor must be contiguous.
     *
     * @param[in] shape Target shape; product of dims must equal `numel()`.
     * @return View with contiguous strides for `shape`.
     * @throws std::invalid_argument if the tensor is a scalar, non-contiguous,
     *         or `numel` would change.
     */
    Tensor<T> reshape(const std::vector<int64_t>& shape) const;

    /**
     * Swap two dimensions by swapping their shape entries and strides (zero-copy).
     *
     * @param[in] dim0 First dimension (negative indices allowed).
     * @param[in] dim1 Second dimension (negative indices allowed).
     * @return View with the two axes exchanged.
     * @throws std::invalid_argument if the tensor is a scalar.
     * @throws std::out_of_range if a dimension is out of bounds.
     */
    Tensor<T> transpose(int64_t dim0, int64_t dim1) const;

    /**
     * Swap the two axes of a rank-2 tensor. Equivalent to `transpose(0, 1)`.
     *
     * @return View with shape `[n, m]` if this tensor is `[m, n]`.
     * @throws std::invalid_argument if rank is not 2.
     */
    Tensor<T> transpose() const;

    /**
     * Expand this tensor to `target_shape` by setting stride 0 on broadcast axes.
     * Zero-copy: elements are not repeated in memory.
     *
     * @param[in] target_shape Compatible shape (NumPy right-alignment rules).
     * @return View with stride 0 on expanded axes.
     * @throws std::invalid_argument if the shapes are not broadcast-compatible.
     */
    Tensor<T> broadcast_to(const std::vector<int64_t>& target_shape) const;

    /**
     * Share storage under a new contiguous shape. Identical to `reshape` here.
     *
     * @param[in] shape Target shape; product of dims must equal `numel()`.
     * @return View with contiguous strides for `shape`.
     * @throws std::invalid_argument if the tensor is a scalar, non-contiguous,
     *         or `numel` would change.
     */
    Tensor<T> view(const std::vector<int64_t>& shape) const;

    /**
     * Collapse dimensions `[start_dim, end_dim]` (inclusive) into one.
     * Requires a contiguous tensor.
     *
     * @param[in] start_dim First collapsed axis (default 0; negative allowed).
     * @param[in] end_dim Last collapsed axis (default -1; negative allowed).
     * @return View with those axes merged.
     * @throws std::invalid_argument if the tensor is a scalar, non-contiguous,
     *         or `start_dim > end_dim` after normalization.
     */
    Tensor<T> flatten(int64_t start_dim = 0, int64_t end_dim = -1) const;

    /**
     * Remove every size-1 dimension. Strides of remaining axes are kept, so
     * this is valid on non-contiguous tensors.
     *
     * @return View with all size-1 axes dropped (scalar if every dim was 1).
     */
    Tensor<T> squeeze() const;

    /**
     * Remove the size-1 dimension at `dim`. No-op (still a view) if that
     * dimension is not size 1.
     *
     * @param[in] dim Axis to drop (negative indices allowed).
     * @return View with that axis removed, or an unchanged view.
     * @throws std::out_of_range if `dim` is out of bounds.
     */
    Tensor<T> squeeze(int64_t dim) const;

    /**
     * Insert a size-1 dimension at `dim`. Valid range is `[-rank-1, rank]`.
     *
     * @param[in] dim Insertion index (negative indices allowed).
     * @return View with a new size-1 axis.
     * @throws std::out_of_range if `dim` is outside `[-rank-1, rank]`.
     */
    Tensor<T> unsqueeze(int64_t dim) const;

    /**
     * Return a tensor whose logical order is packed into a dense row-major buffer.
     * If this tensor is already contiguous with `offset() == 0`, returns `*this`
     * (same `data_ptr_`, no extra graph node). Otherwise allocates `numel()`
     * elements and copies with the stride formula.
     *
     * @return Contiguous tensor with the same shape and logical values.
     */
    Tensor<T> contiguous() const;

    /**
     * View of a slice along `dim`: indices `[start, start+length)`.
     *
     * @param[in] dim Axis to slice (negative indices allowed).
     * @param[in] start First index along `dim`.
     * @param[in] length Number of elements to keep.
     * @return View sharing storage with `*this`.
     * @throws std::invalid_argument if the tensor is scalar or the range is invalid.
     */
    Tensor<T> narrow(int64_t dim, int64_t start, int64_t length) const;

    // ----- math operations -----

    /**
     * Element-wise addition. Inputs must have the same shape (no broadcasting
     * inside the kernel; call `broadcast_to` first). Allocates a new buffer.
     *
     * @param[in] other Right-hand operand.
     * @return New tensor `*this + other`.
     * @throws std::invalid_argument on shape mismatch.
     */
    Tensor<T> add(const Tensor<T>& other) const;

    /**
     * Element-wise negation. Allocates a new buffer.
     *
     * @return New tensor `-(*this)`.
     */
    Tensor<T> neg() const;

    /**
     * Element-wise subtraction. Same-shape inputs; new buffer.
     *
     * @param[in] other Right-hand operand.
     * @return New tensor `*this - other`.
     * @throws std::invalid_argument on shape mismatch.
     */
    Tensor<T> subtract(const Tensor<T>& other) const;

    /**
     * Element-wise multiplication. Same-shape inputs; new buffer.
     *
     * @param[in] other Right-hand operand.
     * @return New tensor `*this * other`.
     * @throws std::invalid_argument on shape mismatch.
     */
    Tensor<T> multiply(const Tensor<T>& other) const;

    /**
     * Element-wise division. Same-shape inputs; new buffer.
     *
     * @param[in] other Right-hand operand.
     * @return New tensor `*this / other`.
     * @throws std::invalid_argument on shape mismatch.
     * @throws std::runtime_error on division by zero.
     */
    Tensor<T> divide(const Tensor<T>& other) const;

    /**
     * Element-wise power with a scalar exponent. New buffer.
     *
     * @param[in] exponent Scalar exponent.
     * @return New tensor `(*this) ** exponent`.
     */
    Tensor<T> power(const T exponent) const;

    /**
     * Element-wise absolute value. New buffer.
     *
     * @return New tensor `| *this |`.
     */
    Tensor<T> abs() const;

    /**
     * Element-wise exponential. New buffer.
     *
     * @return New tensor `exp(*this)`.
     */
    Tensor<T> exp() const;

    /**
     * Element-wise natural logarithm. New buffer.
     *
     * @return New tensor `log(*this)`.
     */
    Tensor<T> log() const;

    /**
     * Element-wise square root. New buffer.
     *
     * @return New tensor `sqrt(*this)`.
     */
    Tensor<T> sqrt() const;

    /**
     * Element-wise sine. New buffer.
     *
     * @return New tensor `sin(*this)`.
     */
    Tensor<T> sin() const;

    /**
     * Element-wise cosine. New buffer.
     *
     * @return New tensor `cos(*this)`.
     */
    Tensor<T> cos() const;

    /**
     * Element-wise tangent. New buffer.
     *
     * @return New tensor `tan(*this)`.
     */
    Tensor<T> tan() const;

    /**
     * Element-wise hyperbolic sine. New buffer.
     *
     * @return New tensor `sinh(*this)`.
     */
    Tensor<T> sinh() const;

    /**
     * Element-wise hyperbolic cosine. New buffer.
     *
     * @return New tensor `cosh(*this)`.
     */
    Tensor<T> cosh() const;

    /**
     * Element-wise hyperbolic tangent. New buffer.
     *
     * @return New tensor `tanh(*this)`.
     */
    Tensor<T> tanh() const;

    /**
     * Element-wise sigmoid, `1 / (1 + exp(-x))`. New buffer.
     *
     * @return New tensor in `(0, 1)`.
     */
    Tensor<T> sigmoid() const;

    /**
     * Element-wise ReLU, `max(0, x)`. New buffer.
     *
     * @return New tensor with negatives zeroed.
     */
    Tensor<T> relu() const;

    /**
     * Element-wise SiLU (swish), `x * sigmoid(x)`. New buffer.
     *
     * @return New tensor.
     */
    Tensor<T> silu() const;

    /**
     * Element-wise GELU (tanh approximation). New buffer.
     *
     * @return New tensor.
     */
    Tensor<T> gelu() const;

    // ----- reduction operations -----

    /**
     * Sum of every element, as a scalar tensor of shape `{}`.
     *
     * @return Scalar tensor.
     * @throws std::invalid_argument if the tensor is empty.
     */
    Tensor<T> sum() const;

    /**
     * Sum along one axis; that axis is removed from the output shape.
     *
     * @param[in] dim Axis to reduce (negative indices allowed).
     * @return Tensor with `dim` dropped.
     * @throws std::invalid_argument if the tensor is a scalar.
     * @throws std::out_of_range if `dim` is out of bounds.
     */
    Tensor<T> sum(int64_t dim) const;

    /**
     * Mean of every element, as a scalar tensor of shape `{}`.
     *
     * @return Scalar tensor.
     * @throws std::invalid_argument if the tensor is empty.
     */
    Tensor<T> mean() const;

    /**
     * Mean along one axis; that axis is removed from the output shape.
     *
     * @param[in] dim Axis to reduce (negative indices allowed).
     * @return Tensor with `dim` dropped.
     * @throws std::invalid_argument if the tensor is a scalar.
     * @throws std::out_of_range if `dim` is out of bounds.
     */
    Tensor<T> mean(int64_t dim) const;

    /**
     * Maximum element as a scalar tensor of shape `{}`. Ties keep the first index.
     *
     * @return Scalar tensor.
     * @throws std::invalid_argument if the tensor is empty.
     */
    Tensor<T> max() const;

    /**
     * Minimum element as a scalar tensor of shape `{}`. Ties keep the first index.
     *
     * @return Scalar tensor.
     * @throws std::invalid_argument if the tensor is empty.
     */
    Tensor<T> min() const;

    /**
     * Softmax along `dim` (max-subtraction for stability). Output shape matches input.
     *
     * @param[in] dim Axis to normalize (negative indices allowed).
     * @return Tensor of the same shape; each slice along `dim` sums to 1.
     * @throws std::invalid_argument if the tensor is a scalar.
     * @throws std::out_of_range if `dim` is out of bounds.
     */
    Tensor<T> softmax(int64_t dim) const;

    // ----- linalg operations -----

    /**
     * Inner product of two 1-D tensors of equal length. Result shape is `{}`.
     *
     * @param[in] other Other vector.
     * @return Scalar tensor.
     * @throws std::invalid_argument if either input is not 1-D or lengths differ.
     */
    Tensor<T> dot(const Tensor<T>& other) const;

    /**
     * Matrix product of two 2-D tensors. Reads through strides, so transposed
     * inputs are correct without calling `contiguous()`.
     *
     * @param[in] other Right-hand matrix of shape `[K, N]` if `*this` is `[M, K]`.
     * @return New contiguous tensor of shape `[M, N]`.
     * @throws std::invalid_argument if ranks are not 2 or inner dims mismatch.
     */
    Tensor<T> matmul(const Tensor<T>& other) const;

    // ----- grad / backward -----

    /**
     * True if this tensor was not produced by an op (`grad_fn_ == nullptr`).
     * User-created tensors are leaves; differentiable `to()` outputs are not.
     *
     * @return True for leaves.
     */
    bool is_leaf() const { return grad_fn_ == nullptr; }

    /**
     * Accumulated gradient of a leaf, or nullptr if backward has not run
     * (or this tensor does not require grad).
     *
     * @return Pointer into `GradStorage`, or nullptr.
     */
    const Tensor<T>* grad() const;

    /**
     * Add `grad` into this leaf's gradient buffer. Engine-internal.
     *
     * @param[in] grad Incoming gradient; shape must match this tensor.
     */
    void accumulate_grad(const Tensor<T>& grad);

    /**
     * Drop the gradient tensor but keep `GradStorage` so later aliases still
     * share it. Call before each backward pass.
     */
    void zero_grad() const;

    /**
     * Run reverse-mode autodiff from this scalar output.
     * Accumulates into `.grad()` of every reachable leaf that requires grad.
     *
     * @return `*this` (unchanged).
     * @throws std::invalid_argument if `numel() != 1` or `grad_fn` is null.
     */
    Tensor<T> backward() const;
};

} // namespace tensor