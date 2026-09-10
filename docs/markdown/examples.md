# Examples

Runnable programs under `examples/`. They compile against the header-only library (`-I src`) and print step logs plus a `FINAL RESULTS` block.

Recipes without a full program: [howto.md](howto.md).

Binaries land in `build/examples/`. Scripts compile with `-O2` (packed kernels SIMD). Override the compiler with `CXX=...` if needed.

Each program enables [logging](logging.md) with `logging::LoggingContext(true)`. Training loops wrap the forward/backward pass in `logging::NoLogContext` so library `contiguous()` messages do not drown out the step logs.

```zsh
# matmul + linear_regression + wine_nn (defaults)
./scripts/run_examples.sh
./scripts/run_examples.sh -n matmul
```

---

## matmul

**Source:** [`examples/matmul.cpp`](../../examples/matmul.cpp)

**Runner:** `./scripts/run_examples.sh -n matmul`

Matrix multiply plus a scalar backward. Builds `A` `{2, 3}` (requires grad) and `B` `{3, 2}`, computes `C = ops::matmul(A, B)`, then `C.sum().backward()` and prints `A.grad`.

```cpp
auto C = ops::matmul(A, B);
auto loss = C.sum();
loss.backward();
const tensor::Tensor<float>* gA = A.grad();
```

Related: [ops.md](ops.md) (`matmul`), [autograd.md](autograd.md).

---

## linear_regression

**Source:** [`examples/linear_regression.cpp`](../../examples/linear_regression.cpp)

**Runner:** `scripts/run_linear_regression.sh`

Fits `y = true_weight * x + true_bias + noise` on synthetic data with `nn::Linear(1, 1)`, `nn::MSELoss`, and `nn::optim::SGD`. Logs MSE / weight / bias every `--log-every` steps; the final block compares learned parameters to the ground truth.

```zsh
./scripts/run_linear_regression.sh
./scripts/run_linear_regression.sh --n-steps 100 --learning-rate 0.02
./scripts/run_linear_regression.sh --true-weight 3 --true-bias 0.5 --noise-std 0.2
./scripts/run_linear_regression.sh --help
```

| Flag | Meaning | Default |
|---|---|---|
| `--n-samples` | Synthetic points | 200 |
| `--n-steps` | SGD steps | 250 |
| `--log-every` | Log period | 25 |
| `--true-weight` | Ground-truth slope | 2.5 |
| `--true-bias` | Ground-truth intercept | -1 |
| `--noise-std` | Gaussian noise std | 0.1 |
| `--learning-rate` | SGD step size | 0.05 |
| `--random-seed` | RNG seed for `x` and noise | 42 |

```cpp
nn::Linear<float> model(1, 1);
nn::MSELoss<float> criterion;
nn::optim::SGD<float> opt(model.parameters(), learning_rate);

for (int step = 1; step <= n_steps; ++step) {
    opt.zero_grad();
    auto pred = model.forward(x);
    auto loss = criterion.forward(pred, y);
    loss.backward();
    opt.step();
}
```

Related: [layers/reference.md](layers/reference.md) (`Linear`, `MSELoss`), [optim/reference.md](optim/reference.md), [optim/mechanism.md](optim/mechanism.md).

---

## wine_nn

**Source:** [`examples/wine_nn.cpp`](../../examples/wine_nn.cpp)

**Runner:** `scripts/run_wine_nn.sh`

Small MLP on the [UCI Wine Quality](https://archive.ics.uci.edu/dataset/186/wine+quality) white-wine set ([Cortez et al., 2009](https://doi.org/10.24432/C56S3T)). Download **`winequality-white.csv`** from that page (not the red-wine file). Eleven physicochemical features; quality scores 3–9 are binned to three classes (low / medium / high). Train/val split, train-set standardization, Adam, dropout, per-epoch shuffle. Reports train loss, validation loss, and validation accuracy.

Architecture: `Linear(11, hidden_1) → ReLU → Dropout → Linear(hidden_1, hidden_2) → ReLU → Dropout → Linear(hidden_2, 3)` with `CrossEntropyLoss` (soft one-hot labels).

Defaults are a small two-hidden MLP (`96 → 48`, ~6k parameters, dropout 0.1, Adam `1e-3`, batch 32). Honest validation accuracy on this 3-class task is typically in the mid-60s; 90%+ figures are usually a different problem (red vs white) or a leaky resample.

The CSV is not in git (`data-example/` is gitignored). After downloading, either place it at the default path or pass `--csv`:

```zsh
mkdir -p data-example/wine-dataset
# copy winequality-white.csv into data-example/wine-dataset/
./scripts/run_wine_nn.sh
./scripts/run_wine_nn.sh --csv /path/to/winequality-white.csv
./scripts/run_wine_nn.sh --n-epochs 80 --report-epochs 10
./scripts/run_wine_nn.sh --hidden-1 64 --hidden-2 32 --dropout 0.2
./scripts/run_wine_nn.sh --help
```

| Flag | Meaning | Default |
|---|---|---|
| `--csv` | Path to `winequality-white.csv` | `data-example/wine-dataset/winequality-white.csv` |
| `--hidden-1` | First hidden width | 96 |
| `--hidden-2` | Second hidden width | 48 |
| `--n-epochs` | Training epochs | 150 |
| `--report-epochs` | Log train loss, val loss, and val accuracy every N epochs (always includes epoch 1 and the last epoch) | 1 |
| `--batch-size` | Mini-batch size | 32 |
| `--learning-rate` | Adam step size | 0.001 |
| `--weight-decay` | Adam L2 coefficient | 1e-4 |
| `--dropout` | Drop probability after each hidden ReLU (`0` disables) | 0.1 |
| `--train-fraction` | Train split in `(0, 1)` | 0.8 |
| `--random-seed` | RNG seed | 42 |

```cpp
class WineNet : public nn::Module<float> {
    nn::Linear<float> fc1;   // (11, hidden_1)
    nn::Linear<float> fc2;   // (hidden_1, hidden_2)
    nn::Linear<float> fc3;   // (hidden_2, 3)
    nn::Dropout<float> drop1;
    nn::Dropout<float> drop2;
    // register_module in the constructor; forward is ReLU + Dropout between the linears
};
```

Related: [layers/mechanism.md](layers/mechanism.md) (`Module`, `register_module`), [layers/reference.md](layers/reference.md) (`CrossEntropyLoss`), [howto.md](howto.md).
