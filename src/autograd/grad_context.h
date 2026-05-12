#pragma once

namespace autograd {

namespace detail {
inline bool& grad_enabled() {
    thread_local bool val = true;
    return val;
}
} // namespace detail

inline bool is_grad_enabled() {
    return detail::grad_enabled();
}

inline void set_grad_enabled(const bool enabled) {
    detail::grad_enabled() = enabled;
}

// RAII guard for grad mode:
// - on construction: remember current mode, then switch to requested mode
// - on destruction: restore previous mode automatically
// This guarantees mode restoration on every scope exit path (normal return or exception).
// grad mode lives only in scope of the guard object
class AutoGradContext {
private:
    bool previous_;

public:
    explicit AutoGradContext(const bool enabled)
        : previous_(is_grad_enabled()) {
        set_grad_enabled(enabled);
    }

    ~AutoGradContext() {
        set_grad_enabled(previous_);
    }

    AutoGradContext(const AutoGradContext&) = delete;
    auto operator=(const AutoGradContext&) -> AutoGradContext& = delete;
};

// RAII guard for disabling gradient computation:
// - on construction: switch to no-grad mode
// - on destruction: restore previous mode automatically
class NoGradContext final : public AutoGradContext {
public:
    NoGradContext() : AutoGradContext(false) {}
};

} // namespace autograd
