#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "autograd/autograd.h"
#include "logging/logger.h"
#include "nn/layers/linear.h"
#include "nn/layers/loss_fn.h"
#include "nn/optim/sgd.h"

namespace {

struct Config {
    int n_samples = 200;
    int n_steps = 250;
    int log_every = 25;
    float true_weight = 2.5f;
    float true_bias = -1.0f;
    float noise_std = 0.1f;
    double learning_rate = 0.05;
    int random_seed = 42;
};

std::string fmt(float v) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(4) << v;
    return oss.str();
}

/**
 * Print CLI flags and defaults to stdout.
 */
void print_usage() {
    const Config d;
    std::cout
        << "Usage: linear_regression [options]\n"
        << "  --n-samples N         number of synthetic points (default " << d.n_samples << ")\n"
        << "  --n-steps N           SGD steps (default " << d.n_steps << ")\n"
        << "  --log-every N         log every N steps (default " << d.log_every << ")\n"
        << "  --true-weight F       ground-truth slope (default " << d.true_weight << ")\n"
        << "  --true-bias F         ground-truth intercept (default " << d.true_bias << ")\n"
        << "  --noise-std F         Gaussian noise std (default " << d.noise_std << ")\n"
        << "  --learning-rate F     SGD step size (default " << d.learning_rate << ")\n"
        << "  --random-seed N       RNG seed for x and noise (default " << d.random_seed << ")\n"
        << "  -h, --help            show this help\n";
}

/**
 * Parse a flag value as a signed integer.
 *
 * @param text Token after the flag.
 * @param flag Flag name, used in the error message.
 * @return Parsed integer.
 *
 * @throws std::invalid_argument if `text` is not an integer.
 */
int parse_int(const std::string& text, const std::string& flag) {
    try {
        size_t idx = 0;
        const int v = std::stoi(text, &idx);
        if (idx != text.size())
            throw std::invalid_argument("trailing characters");
        return v;
    } catch (const std::exception&) {
        throw std::invalid_argument(flag + " expects an integer, got '" + text + "'");
    }
}

/**
 * Parse a flag value as a floating-point number.
 *
 * @param text Token after the flag.
 * @param flag Flag name, used in the error message.
 * @return Parsed double.
 *
 * @throws std::invalid_argument if `text` is not a number.
 */
double parse_double(const std::string& text, const std::string& flag) {
    try {
        size_t idx = 0;
        const double v = std::stod(text, &idx);
        if (idx != text.size())
            throw std::invalid_argument("trailing characters");
        return v;
    } catch (const std::exception&) {
        throw std::invalid_argument(flag + " expects a number, got '" + text + "'");
    }
}

/**
 * Consume the next argv token as a flag value.
 *
 * @param i Current index; incremented to the value token.
 * @param argc Argument count.
 * @param argv Argument vector.
 * @param flag Flag being parsed.
 * @return The value token.
 *
 * @throws std::invalid_argument if no token follows `flag`.
 */
const char* require_value(int& i, int argc, char** argv, const std::string& flag) {
    if (i + 1 >= argc)
        throw std::invalid_argument(flag + " requires a value");
    return argv[++i];
}

/**
 * Parse CLI flags into a `Config`. Unknown flags and missing values throw.
 * `--help` / `-h` print usage and exit 0.
 *
 * @param argc Argument count.
 * @param argv Argument vector.
 * @return Filled config (defaults, then last-wins overrides).
 *
 * @throws std::invalid_argument on bad flags, values, or out-of-range settings.
 */
Config parse_args(int argc, char** argv) {
    Config cfg;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            print_usage();
            std::exit(0);
        }
        if (arg == "--n-samples")
            cfg.n_samples = parse_int(require_value(i, argc, argv, arg), arg);
        else if (arg == "--n-steps")
            cfg.n_steps = parse_int(require_value(i, argc, argv, arg), arg);
        else if (arg == "--log-every")
            cfg.log_every = parse_int(require_value(i, argc, argv, arg), arg);
        else if (arg == "--true-weight")
            cfg.true_weight = static_cast<float>(parse_double(require_value(i, argc, argv, arg), arg));
        else if (arg == "--true-bias")
            cfg.true_bias = static_cast<float>(parse_double(require_value(i, argc, argv, arg), arg));
        else if (arg == "--noise-std")
            cfg.noise_std = static_cast<float>(parse_double(require_value(i, argc, argv, arg), arg));
        else if (arg == "--learning-rate")
            cfg.learning_rate = parse_double(require_value(i, argc, argv, arg), arg);
        else if (arg == "--random-seed")
            cfg.random_seed = parse_int(require_value(i, argc, argv, arg), arg);
        else
            throw std::invalid_argument("unknown flag: " + arg);
    }
    if (cfg.n_samples <= 0)
        throw std::invalid_argument("--n-samples must be > 0");
    if (cfg.n_steps <= 0)
        throw std::invalid_argument("--n-steps must be > 0");
    if (cfg.log_every <= 0)
        throw std::invalid_argument("--log-every must be > 0");
    if (cfg.noise_std < 0.f)
        throw std::invalid_argument("--noise-std must be >= 0");
    if (cfg.learning_rate <= 0.0)
        throw std::invalid_argument("--learning-rate must be > 0");
    return cfg;
}

/**
 * Build synthetic pairs `(x, y)` from `y = true_weight * x + true_bias + noise`.
 * `x` is drawn from a standard normal; noise is i.i.d. N(0, `noise_std`).
 *
 * @param cfg Sample count, ground-truth line, noise std, and RNG seed.
 * @return `{x, y}` with shapes `{n_samples, 1}`.
 */
std::pair<tensor::Tensor<float>, tensor::Tensor<float>> make_synthetic_data(const Config& cfg) {
    auto x = tensor::Tensor<float>::randn(
        {cfg.n_samples, 1}, false, static_cast<uint64_t>(cfg.random_seed));
    auto noise = tensor::Tensor<float>::random_gaussian(
        {cfg.n_samples, 1}, 0.f, cfg.noise_std, false, static_cast<uint64_t>(cfg.random_seed));

    std::vector<float> y_vals(static_cast<size_t>(cfg.n_samples));
    for (int i = 0; i < cfg.n_samples; ++i) {
        const float xi = x.data()[static_cast<size_t>(i)];
        y_vals[static_cast<size_t>(i)] =
            cfg.true_weight * xi + cfg.true_bias + noise.data()[static_cast<size_t>(i)];
    }
    tensor::Tensor<float> y({cfg.n_samples, 1}, y_vals, false);
    return {x, y};
}

} // namespace

int main(int argc, char** argv) {
    Config cfg;
    try {
        cfg = parse_args(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        print_usage();
        return 1;
    }

    logging::LoggingContext logs(true);

    logging::info(
        "linear regression on synthetic data: y = " + fmt(cfg.true_weight) +
        " * x + " + fmt(cfg.true_bias) + " + N(0, " + fmt(cfg.noise_std) + ")");
    logging::info(
        "n=" + std::to_string(cfg.n_samples) +
        "  steps=" + std::to_string(cfg.n_steps) +
        "  lr=" + fmt(static_cast<float>(cfg.learning_rate)) + "  SGD");

    auto [x, y] = make_synthetic_data(cfg);

    nn::Linear<float> model(1, 1);
    nn::MSELoss<float> criterion;
    nn::optim::SGD<float> opt(model.parameters(), cfg.learning_rate);

    logging::info(
        "initial weight=" + fmt(model.weight.data()[0]) +
        "  bias=" + fmt(model.bias->data()[0]));

    float last_loss = 0.f;
    for (int step = 1; step <= cfg.n_steps; ++step) {
        {
            logging::NoLogContext quiet;
            opt.zero_grad();
            auto pred = model.forward(x);
            auto loss = criterion.forward(pred, y);
            loss.backward();
            opt.step();
            last_loss = loss.data()[0];
        }

        if (step % cfg.log_every == 0 || step == 1) {
            logging::info(
                "step " + std::to_string(step) + "/" + std::to_string(cfg.n_steps) +
                "  mse=" + fmt(last_loss) +
                "  w=" + fmt(model.weight.data()[0]) +
                "  b=" + fmt(model.bias->data()[0]));
        }
    }

    const float learned_w = model.weight.data()[0];
    const float learned_b = model.bias->data()[0];

    std::cout << "\n========== FINAL RESULTS ==========\n"
              << "true     w=" << fmt(cfg.true_weight) << "  b=" << fmt(cfg.true_bias) << "\n"
              << "learned  w=" << fmt(learned_w) << "  b=" << fmt(learned_b) << "\n"
              << "final MSE=" << fmt(last_loss) << "\n"
              << "===================================\n";
    return 0;
}
