# Agent notes

## Tests

Use [doctest](tests/doctest/doctest.h). Build and run with `scripts/run_tests.sh` (uses `g++-14`, C++17). Do not invoke the compiler by hand unless the script cannot cover the case.

```zsh
./scripts/run_tests.sh
./scripts/run_tests.sh -n tests/ops/test_elementwise_ops.cpp
./scripts/run_tests.sh -n tests/ops/test_elementwise_ops.cpp -v
```

`-n` takes one or more test `.cpp` files (not `test_main.cpp`). `-v` prints passing assertions. Full command examples live in `README.md`.
