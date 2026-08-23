# Getting Started

EduTorch is a header-only C++17 tensor and automatic differentiation library. It mirrors the core ideas of PyTorch — dynamic computation graphs, stride-based views, and reverse-mode autodiff — in under ~2 000 lines of readable code.

## Requirements

- C++17 compiler (`g++-15` recommended)
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
│   ├── elementwise_ops.h     # add, neg, multiply, sin, relu, ...
│   ├── shape_ops.h           # reshape, transpose, broadcast_to, ...
│   ├── reduction_ops.h       # sum, mean, max, min
│   ├── linalg_ops.h          # dot, matmul
│   └── mixed_dtype_ops.h     # cross-dtype arithmetic
└── autograd/
    ├── autograd.h            # Umbrella include — use this one header
    ├── grad_context.h        # NoGradContext, is_grad_enabled
    ├── node.h                # Node<T> abstract base
    ├── accumulate_grad.h     # Leaf gradient accumulator
    ├── engine.h              # run_backward (topological engine)
    └── backward_ops/         # Concrete backward Node subclasses
```

## Including the library

Always include the umbrella header. It pulls in every sub-header in the correct order and instantiates `tensor_methods.tpp`:

```cpp
#include "autograd/autograd.h"
```

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
auto r = Tensor<float>::randn({4, 4});          // standard normal
auto g = Tensor<float>::random_gaussian({4}, 0.0f, 1.0f, /*seed=*/42);

// scalar (0-dim) tensor
Tensor<float> s({}, {3.14f});
```

## Enabling gradient tracking

Pass `requires_grad = true` to the constructor or factory:

```cpp
Tensor<float> x({3}, {1.0f, 2.0f, 3.0f}, /*requires_grad=*/true);
```

Integer tensors (`Int32`, `Int64`) cannot track gradients — the flag is silently ignored.

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

        // gradient descent (manual, no optimizer yet)
        auto& gw = *w.grad();
        auto& wd = w.data();
        for (size_t i = 0; i < wd.size(); ++i)
            wd[i] -= 0.01f * gw.data()[i];

        // reset
        w.zero_grad();
    }
}
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
| Single dtype per graph | All tensors in one computation graph must share the same `T`. Use `mixed_dtype_ops.h` for cross-type arithmetic (result is a fresh graph). |
| Scalar backward only | `.backward()` requires `numel() == 1`. There is no support for non-scalar loss or custom upstream gradients. |
| Contiguous requirement | `reshape`, `flatten`, and elementwise ops read `data()[i]` directly. Always call them on contiguous tensors. `matmul` and `dot` are stride-safe. |
| No in-place ops | In-place mutation during a backward pass is not detected and will silently corrupt gradients. |
