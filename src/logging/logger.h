#pragma once

#include <iostream>
#include <string>

namespace logging {

namespace detail {
/**
 * Access the thread-local logging-enabled flag. The flag is stored as a
 * `thread_local` bool that defaults to false.
 *
 * @return A mutable reference to this thread's logging-enabled flag.
 */
inline bool& logging_enabled() {
    thread_local bool val = false;
    return val;
}
} // namespace detail

/**
 * Report whether logging is enabled on this thread. Reads the thread-local
 * flag returned by `detail::logging_enabled()`.
 *
 * @return `true` if `info`, `warning`, and `error` emit messages; `false` otherwise.
 */
inline bool is_logging_enabled() {
    return detail::logging_enabled();
}

/**
 * Set whether logging is enabled on this thread. Writes the thread-local flag
 * returned by `detail::logging_enabled()`.
 *
 * @param enabled `true` to emit messages; `false` to make `info`, `warning`, and `error` no-ops.
 */
inline void set_logging_enabled(const bool enabled) {
    detail::logging_enabled() = enabled;
}

class LoggingContext {
private:
    bool previous_;

public:
    /**
     * Enable or disable logging for the lifetime of this object. Saves the
     * current thread-local mode, then calls `set_logging_enabled(enabled)`.
     *
     * @param enabled The logging mode to apply until this object is destroyed.
     */
    explicit LoggingContext(const bool enabled)
        : previous_(is_logging_enabled()) {
        set_logging_enabled(enabled);
    }

    /**
     * Restore the logging mode that was active when this object was constructed.
     * Calls `set_logging_enabled` with the saved previous mode.
     */
    ~LoggingContext() {
        set_logging_enabled(previous_);
    }

    /**
     * Deleted. Copying would duplicate the restore-on-destroy obligation.
     *
     * @param other Unused; this constructor is deleted.
     */
    LoggingContext(const LoggingContext&) = delete;

    /**
     * Deleted. Assigning would duplicate the restore-on-destroy obligation.
     *
     * @param other Unused; this operator is deleted.
     * @return Unused; this operator is deleted.
     */
    auto operator=(const LoggingContext&) -> LoggingContext& = delete;
};

class NoLogContext final : public LoggingContext {
public:
    /**
     * Disable logging for the lifetime of this object. Forwards to
     * `LoggingContext(false)`.
     */
    NoLogContext() : LoggingContext(false) {}
};

/**
 * Write an informational message to stdout. No-ops when logging is disabled;
 * otherwise prints `[INFO] ` plus `message` followed by a newline.
 *
 * @param message The text to print after the `[INFO] ` prefix.
 */
inline void info(const std::string& message) {
    if (!is_logging_enabled())
        return;
    std::cout << "[INFO] " << message << std::endl;
}

/**
 * Write a warning message to stderr. No-ops when logging is disabled;
 * otherwise prints `[WARNING] ` plus `message` followed by a newline.
 *
 * @param message The text to print after the `[WARNING] ` prefix.
 */
inline void warning(const std::string& message) {
    if (!is_logging_enabled())
        return;
    std::cerr << "[WARNING] " << message << std::endl;
}

/**
 * Write an error message to stderr. No-ops when logging is disabled;
 * otherwise prints `[ERROR] ` plus `message` followed by a newline.
 *
 * @param message The text to print after the `[ERROR] ` prefix.
 */
inline void error(const std::string& message) {
    if (!is_logging_enabled())
        return;
    std::cerr << "[ERROR] " << message << std::endl;
}

} // namespace logging
