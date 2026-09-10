# autograd

Header-only C++17 tensors and reverse-mode autograd (PyTorch-like views, layers, and optimizers). No dependencies.

Docs: [getting started](docs/markdown/getting_started.md) · [how-to](docs/markdown/howto.md) · [index](docs/markdown/index.md)

## Requirements

- **C++17** — this is the language the library is written for. Compiles use `-std=c++17`; do not lower it.
- A C++17 compiler (**GCC** or **Clang**). The scripts default to `g++-14` so this repo stays reproducible; that is not a library requirement.
- `-O2` **(or** `-O3`**)** — dense kernels pack views into a contiguous buffer, then loop `data()[i]`. That unit-stride loop is what the optimizer turns into SIMD. An unoptimized (`-O0`) build is correct but leaves that work on the table.
- Nothing to install besides the compiler.

## Quick start

### 1. Clone

```zsh
git clone https://github.com/JPcooldev/autograd.git
cd autograd
```

There is no install step. The compiler must see `src/` (`-I src` from the repo root, or `-I /path/to/autograd/src` from elsewhere).

### 2. Choose a compiler

The **standard** is fixed (`-std=c++17`). The **compiler binary** is yours. Scripts read `CXX` and default to `g++-14`:

```zsh
./scripts/run_tests.sh              # g++-14
CXX=g++-15 ./scripts/run_tests.sh
CXX=clang++ ./scripts/run_examples.sh
```

For your own file, any of `c++`, `g++`, `clang++`, `g++-14`, … is fine as long as it accepts `-std=c++17` and GCC/Clang-style flags.

### 3. Write a program

Save this as `my_program.cpp` in the repo root:

```cpp
#include "autograd/autograd.h"

int main() {
    tensor::Tensor<float> x({2}, {1.f, 2.f});
    tensor::Tensor<float> w({2}, {3.f, 4.f});
    auto loss = x.multiply(w).sum();
    loss.backward();
}
```

`autograd/autograd.h` is the umbrella for tensors, ops, and `backward()`. Add `nn/` headers when you use layers or optimizers; add `logging/logger.h` if you want logs.

### 4. Compile and run

```zsh
c++ -std=c++17 -O2 -I src my_program.cpp -o my_program
./my_program
```

Training loops, `nn::Module`, different layers, and checkpoints: [docs/markdown/howto.md](docs/markdown/howto.md).

## Examples

`examples/` — matmul, linear regression, wine-quality MLP. Details: [docs/markdown/examples.md](docs/markdown/examples.md).

```zsh
./scripts/run_examples.sh
./scripts/run_linear_regression.sh --n-steps 100 --learning-rate 0.02
./scripts/run_wine_nn.sh --n-epochs 80 --report-epochs 10
```



## Tests

Using [doctest](tests/doctest/doctest.h). `scripts/run_tests.sh` builds with C++17 and `-O2` and runs the suite.

```zsh
./scripts/run_tests.sh
./scripts/run_tests.sh -n tests/ops/test_elementwise_ops.cpp
./scripts/run_tests.sh -n tests/ops/test_elementwise_ops.cpp tests/tensor/test_tensor_construction.cpp
./scripts/run_tests.sh -v
CXX=clang++ ./scripts/run_tests.sh -n tests/ops/test_elementwise_ops.cpp -v
```

`-n` takes one or more test `.cpp` files (not `test_main.cpp`). `-v` prints passing assertions.

## Project layout

```
src/
├── tensor/          # Tensor<T>, dtype, accessors
├── ops/             # elementwise, shape, conv, pool, loss, …
├── autograd/        # engine, nodes, umbrella header autograd.h
├── nn/              # Module, layers, optimizers, serialize
└── logging/         # optional logger (not in autograd.h)
```

