#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "autograd/autograd.h"
#include "logging/logger.h"
#include "nn/layers/dropout.h"
#include "nn/layers/linear.h"
#include "nn/layers/loss_fn.h"
#include "nn/module.h"
#include "nn/optim/adam.h"

namespace {

constexpr int n_features = 11;
constexpr int n_classes = 3; // low / medium / high quality

struct Config {
    std::string csv_path = "data-example/wine-dataset/winequality-white.csv";
    int hidden_1 = 96;
    int hidden_2 = 48;
    int n_epochs = 150;
    int report_epochs = 1;
    int batch_size = 32;
    double learning_rate = 1e-3;
    double weight_decay = 1e-4;
    double dropout = 0.1;
    double train_fraction = 0.8;
    uint64_t random_seed = 42;
};

std::string fmt(float v, int prec = 4) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(prec) << v;
    return oss.str();
}

/**
 * Print CLI flags and defaults to stdout.
 */
void print_usage() {
    const Config d;
    std::cout
        << "Usage: wine_nn [options]\n"
        << "  --csv PATH              winequality-white.csv path (default " << d.csv_path << ")\n"
        << "  --hidden-1 N            first hidden width (default " << d.hidden_1 << ")\n"
        << "  --hidden-2 N            second hidden width (default " << d.hidden_2 << ")\n"
        << "  --n-epochs N            training epochs (default " << d.n_epochs << ")\n"
        << "  --report-epochs N       log train/val metrics every N epochs (default " << d.report_epochs << ")\n"
        << "  --batch-size N          mini-batch size (default " << d.batch_size << ")\n"
        << "  --learning-rate F       Adam step size (default " << d.learning_rate << ")\n"
        << "  --weight-decay F        Adam L2 coefficient (default " << d.weight_decay << ")\n"
        << "  --dropout F             hidden dropout probability (default " << d.dropout << ")\n"
        << "  --train-fraction F      train split in (0, 1) (default " << d.train_fraction << ")\n"
        << "  --random-seed N         RNG seed (default " << d.random_seed << ")\n"
        << "  -h, --help              show this help\n";
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
 * Parse a flag value as an unsigned 64-bit integer.
 *
 * @param text Token after the flag.
 * @param flag Flag name, used in the error message.
 * @return Parsed value.
 *
 * @throws std::invalid_argument if `text` is not an unsigned integer.
 */
uint64_t parse_u64(const std::string& text, const std::string& flag) {
    try {
        size_t idx = 0;
        const unsigned long long v = std::stoull(text, &idx);
        if (idx != text.size())
            throw std::invalid_argument("trailing characters");
        return static_cast<uint64_t>(v);
    } catch (const std::exception&) {
        throw std::invalid_argument(flag + " expects an unsigned integer, got '" + text + "'");
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
 * `--help` / `-h` print usage and exit 0. Last occurrence of a flag wins.
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
        if (arg == "--csv")
            cfg.csv_path = require_value(i, argc, argv, arg);
        else if (arg == "--hidden-1")
            cfg.hidden_1 = parse_int(require_value(i, argc, argv, arg), arg);
        else if (arg == "--hidden-2")
            cfg.hidden_2 = parse_int(require_value(i, argc, argv, arg), arg);
        else if (arg == "--n-epochs")
            cfg.n_epochs = parse_int(require_value(i, argc, argv, arg), arg);
        else if (arg == "--report-epochs")
            cfg.report_epochs = parse_int(require_value(i, argc, argv, arg), arg);
        else if (arg == "--batch-size")
            cfg.batch_size = parse_int(require_value(i, argc, argv, arg), arg);
        else if (arg == "--learning-rate")
            cfg.learning_rate = parse_double(require_value(i, argc, argv, arg), arg);
        else if (arg == "--weight-decay")
            cfg.weight_decay = parse_double(require_value(i, argc, argv, arg), arg);
        else if (arg == "--dropout")
            cfg.dropout = parse_double(require_value(i, argc, argv, arg), arg);
        else if (arg == "--train-fraction")
            cfg.train_fraction = parse_double(require_value(i, argc, argv, arg), arg);
        else if (arg == "--random-seed")
            cfg.random_seed = parse_u64(require_value(i, argc, argv, arg), arg);
        else
            throw std::invalid_argument("unknown flag: " + arg);
    }
    if (cfg.hidden_1 <= 0)
        throw std::invalid_argument("--hidden-1 must be > 0");
    if (cfg.hidden_2 <= 0)
        throw std::invalid_argument("--hidden-2 must be > 0");
    if (cfg.n_epochs <= 0)
        throw std::invalid_argument("--n-epochs must be > 0");
    if (cfg.report_epochs <= 0)
        throw std::invalid_argument("--report-epochs must be > 0");
    if (cfg.batch_size <= 0)
        throw std::invalid_argument("--batch-size must be > 0");
    if (cfg.learning_rate <= 0.0)
        throw std::invalid_argument("--learning-rate must be > 0");
    if (cfg.weight_decay < 0.0)
        throw std::invalid_argument("--weight-decay must be >= 0");
    if (cfg.dropout < 0.0 || cfg.dropout >= 1.0)
        throw std::invalid_argument("--dropout must be in [0, 1)");
    if (cfg.train_fraction <= 0.0 || cfg.train_fraction >= 1.0)
        throw std::invalid_argument("--train-fraction must be in (0, 1)");
    return cfg;
}

/**
 * Map a UCI wine quality score in [3, 9] onto 3 class bins.
 * 3–5 → 0 (low), 6 → 1 (medium), 7–9 → 2 (high).
 *
 * @param quality Integer quality column from the CSV.
 * @return Class index in `{0, 1, 2}`.
 *
 * @throws std::invalid_argument if `quality` is outside `[3, 9]`.
 */
int quality_to_class(int quality) {
    if (quality < 3 || quality > 9)
        throw std::invalid_argument("unexpected wine quality: " + std::to_string(quality));
    if (quality <= 5)
        return 0;
    if (quality == 6)
        return 1;
    return 2;
}

struct WineData {
    std::vector<float> features; // row-major, n * n_features
    std::vector<int> labels;
    int n = 0;
};

/**
 * Load `winequality-white.csv` (semicolon-separated, quoted header).
 * Skips the header row and stores 11 physicochemical features plus a 3-class label.
 *
 * @param path Filesystem path to the CSV.
 * @return Packed feature buffer and integer class labels.
 *
 * @throws std::runtime_error if the file cannot be opened or has no data rows.
 * @throws std::invalid_argument if a row does not have 12 columns.
 */
WineData load_wine_csv(const std::string& path) {
    std::ifstream in(path);
    if (!in)
        throw std::runtime_error("failed to open wine dataset: " + path);

    WineData out;
    std::string line;
    if (!std::getline(in, line))
        throw std::runtime_error("wine dataset is empty: " + path);

    while (std::getline(in, line)) {
        if (line.empty())
            continue;
        std::vector<float> cols;
        std::stringstream ss(line);
        std::string cell;
        while (std::getline(ss, cell, ';')) {
            if (!cell.empty() && cell.back() == '\r')
                cell.pop_back();
            cols.push_back(std::stof(cell));
        }
        if (cols.size() != static_cast<size_t>(n_features + 1))
            throw std::invalid_argument(
                "expected 12 columns, got " + std::to_string(cols.size()));

        for (int j = 0; j < n_features; ++j)
            out.features.push_back(cols[static_cast<size_t>(j)]);
        out.labels.push_back(quality_to_class(static_cast<int>(std::lround(cols.back()))));
        ++out.n;
    }
    if (out.n == 0)
        throw std::runtime_error("no data rows in wine dataset: " + path);
    return out;
}

/**
 * Shuffle rows in lockstep and split into train / validation sets.
 *
 * @param data Full loaded dataset.
 * @param train_frac Fraction of rows kept for training.
 * @param seed RNG seed for the permutation.
 * @return `{train, val}` copies of the shuffled split.
 */
std::pair<WineData, WineData> split_train_val(
    const WineData& data,
    double train_frac,
    uint64_t seed
) {
    std::vector<int> order(static_cast<size_t>(data.n));
    std::iota(order.begin(), order.end(), 0);
    std::mt19937 rng(static_cast<std::mt19937::result_type>(seed));
    std::shuffle(order.begin(), order.end(), rng);

    const int n_train = static_cast<int>(std::lround(train_frac * data.n));
    WineData train;
    WineData val;
    train.n = n_train;
    val.n = data.n - n_train;

    auto append_row = [&](WineData& dst, int src_i) {
        const size_t base = static_cast<size_t>(src_i) * n_features;
        dst.features.insert(
            dst.features.end(),
            data.features.begin() + static_cast<std::ptrdiff_t>(base),
            data.features.begin() + static_cast<std::ptrdiff_t>(base + n_features));
        dst.labels.push_back(data.labels[static_cast<size_t>(src_i)]);
    };

    for (int i = 0; i < data.n; ++i) {
        if (i < n_train)
            append_row(train, order[static_cast<size_t>(i)]);
        else
            append_row(val, order[static_cast<size_t>(i)]);
    }
    return {train, val};
}

/**
 * Standardize features with train-set mean / std, then apply the same stats to val.
 * Zero-variance columns are left unscaled (std replaced by 1).
 *
 * @param train Training split; features are overwritten in place.
 * @param val Validation split; features are overwritten in place.
 */
void standardize_inplace(WineData& train, WineData& val) {
    std::vector<float> mean(n_features, 0.f);
    std::vector<float> stddev(n_features, 0.f);
    const float n = static_cast<float>(train.n);
    for (int i = 0; i < train.n; ++i)
        for (int j = 0; j < n_features; ++j)
            mean[static_cast<size_t>(j)] += train.features[static_cast<size_t>(i * n_features + j)];
    for (int j = 0; j < n_features; ++j)
        mean[static_cast<size_t>(j)] /= n;

    for (int i = 0; i < train.n; ++i)
        for (int j = 0; j < n_features; ++j) {
            const float d = train.features[static_cast<size_t>(i * n_features + j)]
                          - mean[static_cast<size_t>(j)];
            stddev[static_cast<size_t>(j)] += d * d;
        }
    for (int j = 0; j < n_features; ++j) {
        stddev[static_cast<size_t>(j)] = std::sqrt(stddev[static_cast<size_t>(j)] / n);
        if (stddev[static_cast<size_t>(j)] < 1e-8f)
            stddev[static_cast<size_t>(j)] = 1.f;
    }

    auto apply = [&](WineData& split) {
        for (int i = 0; i < split.n; ++i)
            for (int j = 0; j < n_features; ++j) {
                const size_t idx = static_cast<size_t>(i * n_features + j);
                split.features[idx] =
                    (split.features[idx] - mean[static_cast<size_t>(j)])
                    / stddev[static_cast<size_t>(j)];
            }
    };
    apply(train);
    apply(val);
}

/**
 * Permute rows of a split in place.
 *
 * @param split Dataset whose features and labels are shuffled together.
 * @param seed RNG seed for the permutation.
 */
void shuffle_rows(WineData& split, uint64_t seed) {
    std::vector<int> order(static_cast<size_t>(split.n));
    std::iota(order.begin(), order.end(), 0);
    std::mt19937 rng(static_cast<std::mt19937::result_type>(seed));
    std::shuffle(order.begin(), order.end(), rng);

    WineData shuffled;
    shuffled.n = split.n;
    shuffled.features.reserve(split.features.size());
    shuffled.labels.reserve(split.labels.size());
    for (int src_i : order) {
        const size_t base = static_cast<size_t>(src_i) * n_features;
        shuffled.features.insert(
            shuffled.features.end(),
            split.features.begin() + static_cast<std::ptrdiff_t>(base),
            split.features.begin() + static_cast<std::ptrdiff_t>(base + n_features));
        shuffled.labels.push_back(split.labels[static_cast<size_t>(src_i)]);
    }
    split = std::move(shuffled);
}

/**
 * Pack features into a `{n, 11}` tensor.
 *
 * @param split Dataset split to convert.
 * @return Feature tensor with `requires_grad = false`.
 */
tensor::Tensor<float> features_tensor(const WineData& split) {
    return tensor::Tensor<float>(
        {split.n, n_features}, split.features, false);
}

/**
 * Pack integer labels into a `{n, 3}` one-hot tensor for `CrossEntropyLoss`.
 *
 * @param split Dataset split to convert.
 * @return Soft-label tensor with `requires_grad = false`.
 */
tensor::Tensor<float> one_hot_tensor(const WineData& split) {
    std::vector<float> y(static_cast<size_t>(split.n * n_classes), 0.f);
    for (int i = 0; i < split.n; ++i)
        y[static_cast<size_t>(i * n_classes + split.labels[static_cast<size_t>(i)])] = 1.f;
    return tensor::Tensor<float>({split.n, n_classes}, y, false);
}

class WineNet : public nn::Module<float> {
public:
    nn::Linear<float> fc1;
    nn::Linear<float> fc2;
    nn::Linear<float> fc3;
    nn::Dropout<float> drop1;
    nn::Dropout<float> drop2;

    /**
     * Construct the three-layer MLP and register its linears and dropouts.
     *
     * @param hidden_1 Width of the first hidden layer.
     * @param hidden_2 Width of the second hidden layer.
     * @param dropout Drop probability applied after each hidden ReLU.
     */
    WineNet(int hidden_1, int hidden_2, double dropout)
        : fc1(n_features, hidden_1),
          fc2(hidden_1, hidden_2),
          fc3(hidden_2, n_classes),
          drop1(dropout),
          drop2(dropout) {
        register_module("fc1", fc1);
        register_module("fc2", fc2);
        register_module("fc3", fc3);
        register_module("drop1", drop1);
        register_module("drop2", drop2);
    }

    /**
     * Run Linear → ReLU → Dropout twice, then a linear classifier.
     *
     * @param x Rank-2 batch `{N, 11}`.
     * @return Logits `{N, 3}`.
     */
    tensor::Tensor<float> forward(const tensor::Tensor<float>& x) {
        auto h1 = drop1.forward(ops::relu(fc1.forward(x)));
        auto h2 = drop2.forward(ops::relu(fc2.forward(h1)));
        return fc3.forward(h2);
    }
};

/**
 * Count argmax(logits) matches against integer labels.
 *
 * @param logits Rank-2 `{N, C}` scores (contiguous).
 * @param labels Length-`N` class indices.
 * @return Number of correct predictions.
 */
int count_correct(const tensor::Tensor<float>& logits, const std::vector<int>& labels) {
    const int n = static_cast<int>(logits.shape()[0]);
    int correct = 0;
    for (int i = 0; i < n; ++i) {
        int best = 0;
        float best_v = static_cast<float>(logits[i][0]);
        for (int c = 1; c < n_classes; ++c) {
            const float v = static_cast<float>(logits[i][c]);
            if (v > best_v) {
                best_v = v;
                best = c;
            }
        }
        if (best == labels[static_cast<size_t>(i)])
            ++correct;
    }
    return correct;
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

    logging::info("wine quality MLP  (" + std::to_string(n_features) + " → "
                  + std::to_string(cfg.hidden_1) + " → " + std::to_string(cfg.hidden_2)
                  + " → " + std::to_string(n_classes) + ")");
    logging::info("loading " + cfg.csv_path);

    WineData raw = load_wine_csv(cfg.csv_path);
    auto [train, val] = split_train_val(raw, cfg.train_fraction, cfg.random_seed);
    standardize_inplace(train, val);

    auto x_val = features_tensor(val);
    auto y_val = one_hot_tensor(val);

    WineNet model(cfg.hidden_1, cfg.hidden_2, cfg.dropout);
    nn::CrossEntropyLoss<float> criterion;
    nn::optim::Adam<float> opt(
        model.parameters(),
        cfg.learning_rate,
        0.9,
        0.999,
        1e-8,
        cfg.weight_decay
    );

    logging::info(
        "rows=" + std::to_string(raw.n) +
        "  train=" + std::to_string(train.n) +
        "  val=" + std::to_string(val.n) +
        "  classes=low/medium/high  epochs=" + std::to_string(cfg.n_epochs) +
        "  report_epochs=" + std::to_string(cfg.report_epochs) +
        "  batch=" + std::to_string(cfg.batch_size) +
        "  adam lr=" + fmt(static_cast<float>(cfg.learning_rate)) +
        "  dropout=" + fmt(static_cast<float>(cfg.dropout)) +
        "  n_parameters=" + std::to_string(model.num_parameters())
    );

    float last_train_loss = 0.f;
    float last_val_loss = 0.f;
    float last_val_acc = 0.f;

    for (int epoch = 1; epoch <= cfg.n_epochs; ++epoch) {
        shuffle_rows(train, cfg.random_seed + static_cast<uint64_t>(epoch));
        auto x_train = features_tensor(train);
        auto y_train = one_hot_tensor(train);

        model.train();
        float epoch_loss = 0.f;
        int n_batches = 0;

        {
            logging::NoLogContext quiet;
            for (int start = 0; start < train.n; start += cfg.batch_size) {
                const int len = std::min(cfg.batch_size, train.n - start);
                auto xb = x_train.narrow(0, start, len);
                auto yb = y_train.narrow(0, start, len);

                opt.zero_grad();
                auto logits = model.forward(xb);
                auto loss = criterion.forward(logits, yb);
                loss.backward();
                opt.step();

                epoch_loss += loss.data()[0];
                ++n_batches;
            }
        }
        last_train_loss = epoch_loss / static_cast<float>(n_batches);

        const bool report = epoch == 1
            || epoch == cfg.n_epochs
            || epoch % cfg.report_epochs == 0;
        if (!report)
            continue;

        model.eval();
        {
            logging::NoLogContext quiet;
            autograd::NoGradContext no_grad;
            auto logits = model.forward(x_val);
            auto vloss = criterion.forward(logits, y_val);
            last_val_loss = vloss.data()[0];
            const int correct = count_correct(logits, val.labels);
            last_val_acc = static_cast<float>(correct) / static_cast<float>(val.n);
        }

        logging::info(
            "epoch " + std::to_string(epoch) + "/" + std::to_string(cfg.n_epochs) +
            "  train_loss=" + fmt(last_train_loss) +
            "  val_loss=" + fmt(last_val_loss) +
            "  val_acc=" + fmt(last_val_acc * 100.f, 2) + "%");
    }

    std::cout << "\n========== FINAL RESULTS ==========\n"
              << "train_loss=" << fmt(last_train_loss) << "\n"
              << "val_loss=" << fmt(last_val_loss) << "\n"
              << "val_acc=" << fmt(last_val_acc * 100.f, 2) << "%\n"
              << "===================================\n";
    return 0;
}
