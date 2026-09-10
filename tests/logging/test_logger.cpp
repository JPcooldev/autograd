/*
 * logging::info / warning / error and the RAII enable/disable guards.
 *
 * - logging off by default
 * - set_logging_enabled
 * - info → stdout; warning/error → stderr
 * - all three silent when disabled
 * - LoggingContext restores the previous mode
 * - NoLogContext disables for its lifetime
 * - nested LoggingContext restores each previous mode
 */

#include <iostream>
#include <sstream>
#include <string>

#include "../doctest/doctest.h"

#include "../../src/logging/logger.h"

namespace {

struct StreamCapture {
    std::ostream& stream;
    std::streambuf* previous;
    std::ostringstream buffer;

    explicit StreamCapture(std::ostream& s)
        : stream(s), previous(s.rdbuf(buffer.rdbuf())) {}

    ~StreamCapture() {
        stream.rdbuf(previous);
    }

    std::string str() const { return buffer.str(); }
};

} // namespace

TEST_CASE("logging is disabled by default") {
    CHECK(!logging::is_logging_enabled());
}

TEST_CASE("set_logging_enabled toggles the global flag") {
    const bool previous = logging::is_logging_enabled();
    logging::set_logging_enabled(false);
    CHECK(!logging::is_logging_enabled());
    logging::set_logging_enabled(true);
    CHECK(logging::is_logging_enabled());
    logging::set_logging_enabled(previous);
}

TEST_CASE("info writes to stdout when logging is enabled") {
    logging::LoggingContext guard(true);
    StreamCapture capture(std::cout);
    logging::info("hello");
    CHECK(capture.str() == "[INFO] hello\n");
}

TEST_CASE("warning and error write to stderr when logging is enabled") {
    logging::LoggingContext guard(true);
    StreamCapture capture(std::cerr);
    logging::warning("watch out");
    logging::error("failed");
    CHECK(capture.str() == "[WARNING] watch out\n[ERROR] failed\n");
}

TEST_CASE("info is silent when logging is disabled") {
    logging::LoggingContext guard(false);
    StreamCapture capture(std::cout);
    logging::info("should not appear");
    CHECK(capture.str().empty());
}

TEST_CASE("warning and error are silent when logging is disabled") {
    logging::LoggingContext guard(false);
    StreamCapture capture(std::cerr);
    logging::warning("should not appear");
    logging::error("should not appear");
    CHECK(capture.str().empty());
}

TEST_CASE("LoggingContext restores the previous mode on scope exit") {
    CHECK(!logging::is_logging_enabled());
    {
        logging::LoggingContext guard(true);
        CHECK(logging::is_logging_enabled());
    }
    CHECK(!logging::is_logging_enabled());
}

TEST_CASE("NoLogContext disables logging for its lifetime") {
    logging::LoggingContext enabled(true);
    StreamCapture capture(std::cout);
    {
        logging::NoLogContext guard;
        CHECK(!logging::is_logging_enabled());
        logging::info("silenced");
    }
    CHECK(logging::is_logging_enabled());
    CHECK(capture.str().empty());
    logging::info("restored");
    CHECK(capture.str() == "[INFO] restored\n");
}

TEST_CASE("nested LoggingContext restores each previous mode") {
    CHECK(!logging::is_logging_enabled());
    {
        logging::LoggingContext outer(true);
        CHECK(logging::is_logging_enabled());
        {
            logging::LoggingContext inner(false);
            CHECK(!logging::is_logging_enabled());
        }
        CHECK(logging::is_logging_enabled());
    }
    CHECK(!logging::is_logging_enabled());
}
