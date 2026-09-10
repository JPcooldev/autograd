# Getting Started

**autograd** is a header-only C++17 tensor and automatic differentiation library. It mirrors the core ideas of PyTorch — dynamic computation graphs, stride-based views, and reverse-mode autodiff.

A walkthrough with training loops, `Module`, conv, and checkpoints is in [howto.md](howto.md). All pages: [index.md](index.md).

## Requirements

- **C++17** compiler (GCC or Clang). Scripts default to `g++-14`; pass another binary with `CXX=clang++` / `CXX=g++-15`.
- Compile with **`-O2`** (or `-O3`). Kernels pack to a dense `data()[i]` loop so the optimizer can emit SIMD; `-O0` is correct but slow. Example and test scripts already pass `-O2`.
- No external dependencies — the library is entirely header-only

## Project layout

```
src/
├── tensor/
│   ├── dtype.h               # Dtype enum and thread-local default
│   ├── tensor_accessor.h     # Chained operator[] accessor
│   ├── tensor.h              # Tensor<T> class declaration
│   └── tensor_methods.tpp    # Template method bodies
├── ops/
│   ├── elementwise_ops.h     # add, neg, multiply, sin, relu, silu, …
│   ├── shape_ops.h           # reshape, transpose, narrow, cat, …
│   ├── reduction_ops.h       # sum, mean, max, min, softmax
│   ├── linalg_ops.h          # dot, matmul
│   ├── embedding_ops.h       # embedding lookup
│   ├── conv_ops.h            # conv / conv_transpose 1d/2d/3d
│   ├── pool_ops.h            # max/avg/global pool
│   ├── dropout_ops.h         # inverted dropout
│   ├── loss_ops.h            # l1, mse, nll, cross-entropy, bce, kl
│   └── mixed_dtype_ops.h     # cross-dtype two-tensor ops
├── autograd/
│   ├── autograd.h            # Umbrella include for tensor + core ops
│   ├── grad_context.h        # NoGradContext, is_grad_enabled
│   ├── node.h                # Node<T> abstract base
│   ├── engine.h              # run_backward (topological engine)
│   └── backward_ops/         # Concrete backward Node subclasses
├── nn/
│   ├── module.h              # Module: register_module / named_parameters
│   ├── serialize.h           # save / load AGCK checkpoints
│   ├── layers/               # Linear, Embedding, Conv, Norm, RNN, …
│   └── optim/                # SGD, Adam, AdamW
└── logging/
    └── logger.h              # info/warning/error, LoggingContext
```

## Including the library

For tensors, most `ops::` functions, and `backward()`, include the umbrella header. It instantiates `tensor_methods.tpp`:

```cpp
#include "autograd/autograd.h"
```

That pull includes elementwise, shape, embedding, mixed-dtype (and thus linalg, conv, loss), plus reduction via the `.tpp`. It does **not** include pool, dropout, `nn/`, or logging — include those headers yourself.

> **Do not** include `tensor/tensor.h` alone for code that calls methods implemented in `tensor_methods.tpp` (most math and grad-related methods). Those bodies live in a separate `.tpp` file that is only included through `autograd.h`.

## Creating tensors

All tensors are `tensor::Tensor<T>` where `T` is one of `float`, `double`, `int32_t`, or `int64_t`.

```cpp
using tensor::Tensor;

// from data + shape
Tensor<float> a({2, 3}, {1, 2, 3, 4, 5, 6});

// factory constructors
auto z = Tensor<float>::zeros({3, 4});
auto o = Tensor<float>::ones({2, 2});
auto r = Tensor<float>::randn({4, 4});          // standard normal, requires_grad
auto g = Tensor<float>::random_gaussian({4}, 0.0f, 1.0f, true, /*seed=*/42);

// scalar (0-dim) tensor
Tensor<float> s({}, {3.14f});
```

Integer tensors (`Int32`, `Int64`) cannot track gradients — the flag is silently ignored.

Factories default to `requires_grad = true` for floating `T` (`randint` is always off). Pass `false` when you do not need a graph:

```cpp
Tensor<float> x({3}, {1.0f, 2.0f, 3.0f}, /*requires_grad=*/true);
auto z = Tensor<float>::zeros({3}, /*requires_grad=*/false);
```

## Running a forward pass

Use free functions in the `ops::` namespace directly, or call the equivalent methods on a `Tensor`:

```cpp
#include "autograd/autograd.h"

Tensor<float> x({2}, {2.0f, 3.0f}, true);
Tensor<float> w({2}, {0.5f, 1.0f}, true);

// free-function style
auto z = ops::multiply(x, w);          // [1.0, 3.0]
auto loss = ops::sum(z);               // scalar {4.0}

// method style (same result)
auto loss2 = x.multiply(w).sum();
```

## Running the backward pass

Call `.backward()` on a **scalar** output tensor. All leaf tensors with `requires_grad = true` will have their `.grad()` populated.

```cpp
loss.backward();

// read gradients
auto gx = x.grad();   // pointer to Tensor<float>
auto gw = w.grad();   // pointer to Tensor<float>
```

Multiple calls to `backward()` **accumulate** gradients. Call `.zero_grad()` on each leaf before the next iteration:

```cpp
x.zero_grad();
w.zero_grad();
```

## Disabling gradient tracking

Use `NoGradContext` for inference or when building intermediate tensors that should not be tracked:

```cpp
{
    autograd::NoGradContext no_grad;
    auto y = ops::add(x, w);   // grad_fn is null, no graph recorded
}
```

Or query the state:

```cpp
bool tracking = autograd::is_grad_enabled();   // true by default
```

## Logging

Logging is **off** by default. Include `logging/logger.h` (not part of the autograd umbrella) and enable it with a RAII guard:

```cpp
#include "logging/logger.h"

{
    logging::LoggingContext on(true);
    logging::info("forward pass");
    logging::warning("check this value");
    logging::error("something failed");
}
```

`NoLogContext` silences logs for a scope even if logging was already on. Full API: [logging.md](logging.md).

## A minimal training loop example

```cpp
#include "autograd/autograd.h"

int main() {
    using tensor::Tensor;

    // leaf parameters
    Tensor<float> w({2, 2}, {0.1f, 0.2f, 0.3f, 0.4f}, true);
    Tensor<float> x({2, 1}, {1.0f, 1.0f}, false);

    for (int step = 0; step < 10; ++step) {
        // forward
        auto pred = ops::matmul(w, x);          // 2×1
        auto loss = ops::sum(ops::power(pred, 2.0f));   // scalar MSE proxy

        // backward
        loss.backward();

        // gradient descent (manual, no optimizer — in-place after backward is OK)
        auto& gw = *w.grad();
        auto& wd = w.data();
        for (size_t i = 0; i < wd.size(); ++i)
            wd[i] -= 0.01f * gw.data()[i];

        // reset
        w.zero_grad();
    }
}
```

Full programs (matmul, linear regression, a small wine-quality MLP) live under `examples/` and are documented in [examples.md](examples.md). Recipes: [howto.md](howto.md).

To persist weights, include `nn/serialize.h`. `save` writes each named tensor's **shape, dtype, and packed values**; `load` copies into an already-built model of the same architecture. Strict mode (default) throws if names or shapes disagree; `load(model, path, /*strict=*/false)` skips missing/extra keys.

```cpp
#include "nn/serialize.h"

nn::save(model, "model.agck");
nn::load(model, "model.agck");
```

```zsh
./scripts/run_examples.sh
./scripts/run_linear_regression.sh --n-steps 100 --learning-rate 0.02
./scripts/run_wine_nn.sh --n-epochs 80 --report-epochs 10
```

## Running the tests

```zsh
# run all tests
./scripts/run_tests.sh

# run a specific test file
./scripts/run_tests.sh -n tests/ops/test_elementwise_ops.cpp

# multiple files
./scripts/run_tests.sh -n tests/ops/test_elementwise_ops.cpp tests/tensor/test_tensor_construction.cpp

# verbose output (prints passing assertions)
./scripts/run_tests.sh -v
```

The test suite uses the [doctest](https://github.com/doctest/doctest) framework. Test sources live under `tests/`.

## Important constraints

| Constraint | Details |
|---|---|
| Single dtype per graph | All tensors in one computation graph must share the same `T`. Use `mixed_dtype_ops.h` for cross-type two-tensor ops (result is a fresh graph). |
| Scalar backward only | `.backward()` requires `numel() == 1`. There is no support for non-scalar loss or custom upstream gradients. |
| Contiguous packing | Dense kernels pack at the start (`contiguous()`), then loop `data()[i]` (SIMD-friendly). Shape ops stay views. Not a strided wrapper around the buffer, and not a gather in every kernel — see [tensor/mechanism.md](tensor/mechanism.md#why-dense-kernels-pack-at-the-door). `view` still requires a contiguous source. `matmul` is stride-safe. |
| No in-place ops | In-place mutation during a backward pass is not detected and will silently corrupt gradients. |
