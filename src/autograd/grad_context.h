#pragma once

namespace autograd {

namespace detail {

/**
 * Access the thread-local gradient-enabled flag.
 * Returns a reference so callers can read or assign the current mode.
 *
 * @return Reference to the thread-local flag (default true).
 */
inline bool& grad_enabled() {
    thread_local bool val = true;
    return val;
}

} // namespace detail

/**
 * Query whether gradient tracking is currently enabled on this thread.
 *
 * @return True if ops may attach `grad_fn` nodes.
 */
inline bool is_grad_enabled() {
    return detail::grad_enabled();
}

/**
 * Set whether gradient tracking is enabled on this thread.
 *
 * @param enabled New value of the thread-local flag.
 */
inline void set_grad_enabled(const bool enabled) {
    detail::grad_enabled() = enabled;
}

/**
 * RAII guard for gradient mode.
 * On construction, stores the current mode and switches to `enabled`;
 * on destruction, restores the previous mode (including on exceptions).
 */
class AutoGradContext {
private:
    bool previous_;

public:
    /**
     * Enter a scope with gradient tracking set to `enabled`.
     *
     * @param enabled Requested gradient mode for the lifetime of this guard.
     */
    explicit AutoGradContext(const bool enabled)
        : previous_(is_grad_enabled()) {
        set_grad_enabled(enabled);
    }

    /**
     * Restore the gradient mode captured at construction.
     */
    ~AutoGradContext() {
        set_grad_enabled(previous_);
    }

    /**
     * Deleted copy constructor; gradient mode is scoped to a unique guard.
     *
     * @param other Unused; this constructor is deleted.
     */
    AutoGradContext(const AutoGradContext&) = delete;

    /**
     * Deleted copy assignment; gradient mode is scoped to a unique guard.
     *
     * @return Not used; this operator is deleted.
     */
    auto operator=(const AutoGradContext&) -> AutoGradContext& = delete;
};

/**
 * RAII guard that disables gradient computation for its lifetime.
 * Forwards to `AutoGradContext(false)`.
 */
class NoGradContext final : public AutoGradContext {
public:
    /**
     * Disable gradient tracking until this object is destroyed.
     */
    NoGradContext() : AutoGradContext(false) {}
};

} // namespace autograd
