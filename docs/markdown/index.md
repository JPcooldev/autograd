# Documentation

Header-only C++17 **autograd** library. Start here:

1. [Getting started](getting_started.md) — include path, first tensor, first backward
2. [How-to](howto.md) — short recipes (train a linear model, stack a `Module`, conv, checkpoints)

## Reference

| Page | What it covers |
|---|---|
| [Tensor](tensor.md) | Overview: dtypes, constructors, views, autograd on a tensor |
| [Tensor methods](tensor/methods.md) | Full `tensor::Tensor<T>` API |
| [Tensor internals](tensor/mechanism.md) | Storage, strides, `alias` vs `from_view`, packing |
| [Operations](ops.md) | `ops::` free functions (elementwise, shape, conv, pool, loss, …) |
| [Autograd](autograd.md) | Graph, `NoGradContext`, engine, backward nodes |
| [Layer reference](layers/reference.md) | `nn::Layer` / `nn::Module` and every layer card |
| [Layer internals](layers/mechanism.md) | Named parameters, checkpoints, conv/RNN/dropout/BatchNorm |
| [Optimizers](optim/reference.md) | SGD / Adam / AdamW constructors |
| [Optimizer internals](optim/mechanism.md) | How `step()` and `zero_grad()` use `.grad()` |
| [Optimizer math](optim/theory.md) | Update formulas |
| [Logging](logging.md) | `logging::info` / `LoggingContext` |
| [Examples](examples.md) | Programs under `examples/` |
