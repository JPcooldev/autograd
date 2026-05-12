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
- casting methods: (float32 for now)
    - to()
    - float32()
    - float64()
    - int32()
    - int64()
    - bool_()
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
    - sigmoid()
    - relu()
- reduction operations:
    - sum()
    - sum(dim)
    - mean()
    - mean(dim)
    - max()
    - min()
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

#include <cstdint>     // int64_t, uint64_t
#include <limits>      // std::numeric_limits
#include <memory>      // std::shared_ptr
#include <optional>    // std::optional, std::nullopt
#include <random>      // std::mt19937_64, distributions
#include <stdexcept>   // std::invalid_argument, std::overflow_error
#include <type_traits> // std::is_same
#include <utility>     // std::move
#include <vector>      // std::vector

#include "dtype.h"
#include "tensor_accessor.h"

namespace autograd {
template <typename T>
class Node;
template <typename T>
class AccumulateGrad;
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
    // AccumulateGrad node for this leaf (null for non-leaf / no-grad tensors).
    // Created lazily on first call to ensure_accumulate_grad_fn().
    mutable std::shared_ptr<autograd::Node<T>> accumulate_grad_fn_;

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
        accumulate_grad_fn_(nullptr)
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
        accumulate_grad_fn_(nullptr)
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
        accumulate_grad_fn_(nullptr)
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
        accumulate_grad_fn_(nullptr)
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
    // `source` without copying any data. The alias has requires_grad=false and
    // no grad_fn / accumulate_grad_fn, so it is inert with respect to autograd.
    //
    // This is used by Node::snapshot_tensor() for leaf tensors (parameters,
    // user inputs): their storage is kept alive by the caller, so a shared
    // reference is safe and avoids the O(N) copy cost of a full snapshot.
    //
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
    static Tensor<T> alias(const Tensor<T>& source) {
        // increase ref count for shared pointers (data_ptr_, base_ptr_, grad_fn_ and accumulate_grad_fn_)
        // deep copy metadata
        Tensor<T> t(source);
        t.requires_grad_ = false;
        // decrease ref count by setting it to nullptr (disconnect from computation graph)
        t.grad_fn_ = nullptr;
        t.accumulate_grad_fn_ = nullptr;
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
        view.grad_fn_      = std::move(grad_fn);
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

    // ----- accessors -----

    int64_t rank() const {
        return static_cast<int64_t>(shape_.size());
    }

    int64_t numel() const {
        return compute_numel(shape_);
    }

    const std::vector<int64_t>& shape() const {
        return shape_;
    }

    int64_t ndim() const {
        return shape_.size();
    }

    const std::vector<int64_t>& strides() const {
        return strides_;
    }

    int64_t offset() const {
        return offset_;
    }

    Dtype dtype() const {
        return dtype_;
    }

    bool requires_grad() const {
        return requires_grad_;
    }

    bool is_view() const {
        return base_ptr_ != nullptr;
    }

    bool is_contiguous() const {
        return strides_ == compute_contiguous_strides(shape_);
    }

    const std::shared_ptr<autograd::Node<T>>& grad_fn() const {
        return grad_fn_;
    }

    // Storage accessors are provided as const/non-const references
    const std::vector<T>& data() const {
        return *data_ptr_;
    }

    std::vector<T>& data() {
        return *data_ptr_;
    }

    // Multi-dimensional element access via chained operator[].
    // Usage: tensor[i][j][k]
    // The Tensor acts as the first-level accessor; each operator[] returns a
    // TensorAccessor that consumes one further dimension.
    TensorAccessor<T> operator[](int64_t idx) {
        return TensorAccessor<T>(
            data_ptr_->data() + offset_,
            shape_.data(),
            strides_.data(),
            rank()
        )[idx];
    }

    TensorAccessor<const T> operator[](int64_t idx) const {
        return TensorAccessor<const T>(
            data_ptr_->data() + offset_,
            shape_.data(),
            strides_.data(),
            rank()
        )[idx];
    }

    // ----- casting methods -----

    // Cast all elements to a different scalar type.
    // The returned tensor is always a non-grad leaf (cross-type autograd not yet supported).
    template <typename U>
    Tensor<U> to() const;

    // ----- shape operations -----

    // reshape
    Tensor<T> reshape(const std::vector<int64_t>& shape) const;

    // transpose
    Tensor<T> transpose(int64_t dim0, int64_t dim1) const;

    // transpose matrix
    Tensor<T> transpose() const;

    // broadcast_to: returns a zero-copy stride-0 view expanded to target_shape
    Tensor<T> broadcast_to(const std::vector<int64_t>& target_shape) const;

    // view: shares storage (requires contiguous), identical to reshape in this codebase
    Tensor<T> view(const std::vector<int64_t>& shape) const;

    // flatten dims [start_dim, end_dim] inclusive into one dim (requires contiguous)
    Tensor<T> flatten(int64_t start_dim = 0, int64_t end_dim = -1) const;

    // squeeze: remove all size-1 dimensions
    Tensor<T> squeeze() const;

    // squeeze: remove the size-1 dimension at dim (no-op if size != 1)
    Tensor<T> squeeze(int64_t dim) const;

    // unsqueeze: insert a new size-1 dimension at dim; valid range [-rank-1, rank]
    Tensor<T> unsqueeze(int64_t dim) const;

    // ----- math operations -----
    Tensor<T> add(const Tensor<T>& other) const;
    Tensor<T> neg() const;
    Tensor<T> subtract(const Tensor<T>& other) const;
    Tensor<T> multiply(const Tensor<T>& other) const;
    Tensor<T> divide(const Tensor<T>& other) const;
    Tensor<T> power(const T exponent) const;
    Tensor<T> abs() const;
    Tensor<T> exp() const;
    Tensor<T> log() const;
    Tensor<T> sin() const;
    Tensor<T> cos() const;
    Tensor<T> tan() const;
    Tensor<T> sigmoid() const;
    Tensor<T> relu() const;

    // ----- reduction operations -----
    Tensor<T> sum() const;
    Tensor<T> sum(int64_t dim) const;
    Tensor<T> mean() const;
    Tensor<T> mean(int64_t dim) const;
    Tensor<T> max() const;
    Tensor<T> min() const;

    // ----- linalg operations -----
    Tensor<T> dot(const Tensor<T>& other) const;
    Tensor<T> matmul(const Tensor<T>& other) const;

    // ----- grad / backward -----
    bool is_leaf() const { return grad_fn_ == nullptr; }
    const Tensor<T>* grad() const;
    void zero_grad() const;
    const std::shared_ptr<autograd::Node<T>>& ensure_accumulate_grad_fn() const;

    // ----- backward -----
    Tensor<T> backward() const;
};

} // namespace tensor