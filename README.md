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

## Tests
- use `doctest` framework. its docs are available [here](doctest-docs)
- `scripts/run_tests.sh` builds with `g++-14` (C++17) and runs the suite

```zsh
# run all tests
./scripts/run_tests.sh

# run specific test files
./scripts/run_tests.sh -n tests/ops/test_elementwise_ops.cpp
./scripts/run_tests.sh -n tests/ops/test_elementwise_ops.cpp tests/tensor/test_tensor_construction.cpp

# verbose mode (prints passing assertions)
./scripts/run_tests.sh -v
./scripts/run_tests.sh -n tests/ops/test_elementwise_ops.cpp -v
```