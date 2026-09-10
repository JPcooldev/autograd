#pragma once

namespace tensor {

enum class Dtype {
    Float32,
    Float64,
    Int32,
    Int64
};

namespace detail {
/**
 * Returns a reference to this thread's default dtype storage.
 * The value is initialized to Float32 on first use per thread.
 *
 * @return Mutable reference to the thread-local default dtype.
 */
inline Dtype& default_dtype() {
    thread_local Dtype value = Dtype::Float32;
    return value;
}
} // namespace detail

/**
 * Reads the thread-local default dtype used when a dtype is not specified.
 * Delegates to `detail::default_dtype()`.
 *
 * @return The current default dtype (Float32 after `reset_default_dtype()`).
 */
inline Dtype get_default_dtype() {
    return detail::default_dtype();
}

/**
 * Sets the thread-local default dtype.
 * Writes through `detail::default_dtype()`.
 *
 * @param dtype The dtype to store as the default.
 */
inline void set_default_dtype(const Dtype dtype) {
    detail::default_dtype() = dtype;
}

/**
 * Restores the thread-local default dtype to Float32.
 * Calls `set_default_dtype(Dtype::Float32)`.
 */
inline void reset_default_dtype() {
    set_default_dtype(Dtype::Float32);
}

} // namespace tensor
