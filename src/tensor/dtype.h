#pragma once

namespace tensor {

enum class Dtype {
    Float32,
    Float64,
    Int32,
    Int64
};

namespace detail {
inline Dtype& default_dtype() {
    thread_local Dtype value = Dtype::Float32;
    return value;
}
} // namespace detail

inline Dtype get_default_dtype() {
    return detail::default_dtype();
}

inline void set_default_dtype(const Dtype dtype) {
    detail::default_dtype() = dtype;
}

inline void reset_default_dtype() {
    set_default_dtype(Dtype::Float32);
}

} // namespace tensor