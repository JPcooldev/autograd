# How-to

Short recipes. Compile with `-std=c++17 -O2 -I src` (`-O2` so packed kernels can SIMD). Scripts default to `g++-14`; override with `CXX=clang++` (see the [README](../../README.md)). Full programs: [examples.md](examples.md). API details: [index.md](index.md).

```cpp
#include "autograd/autograd.h"          // tensors, most ops, backward
#include "nn/layers/linear.h"           // nn::Linear
#include "nn/layers/loss_fn.h"          // nn::MSELoss, …
#include "nn/layers/convolutional.h"    // nn::Conv2d
#include "nn/layers/pooling.h"
#include "nn/module.h"
#include "nn/optim/sgd.h"
#include "nn/optim/adam.h"
#include "nn/serialize.h"
```

`autograd.h` does **not** include `nn/` or `logging/`. Pool and dropout free functions need `ops/pool_ops.h` and `ops/dropout_ops.h` if you call them without a layer header.

---

## Scalar backward on a tensor

Factories default to `requires_grad = true` for floating `T`. `backward()` only works on a scalar (`numel() == 1`).

```cpp
#include "autograd/autograd.h"

using tensor::Tensor;

Tensor<float> x({2}, {1.f, 2.f});          // requires_grad = true
Tensor<float> w({2}, {3.f, 4.f});

auto y = ops::multiply(x, w);              // or x.multiply(w)
auto loss = y.sum();                       // scalar
loss.backward();

const Tensor<float>* gx = x.grad();        // {3, 4}
x.zero_grad();                             // before the next step
```

Integer tensors never track gradients. There is no `operator+`; use `ops::add` / `.add`.

---

## Train `nn::Linear` with SGD

Same loop as [examples/linear_regression.cpp](../../examples/linear_regression.cpp).

```cpp
#include "autograd/autograd.h"
#include "nn/layers/linear.h"
#include "nn/layers/loss_fn.h"
#include "nn/optim/sgd.h"

nn::Linear<float> model(1, 1);
nn::MSELoss<float> criterion;
nn::optim::SGD<float> opt(model.parameters(), /*lr=*/0.05);

for (int step = 0; step < 100; ++step) {
    opt.zero_grad();
    auto pred = model.forward(x);          // x shape {N, 1}
    auto loss = criterion.forward(pred, y);
    loss.backward();
    opt.step();                            // in-place on model.weight / bias
}
```

`parameters()` is a `vector<Tensor<T>*>`. Those tensors must outlive the optimizer. Call `zero_grad()` every iteration; Adam moments are **not** cleared by it.

---

## Stack layers with `nn::Module`

There is no `Sequential`. Subclass `Module`, store layers as members, and `register_module` in the constructor so `parameters()` / `save` see them. Activations are ops, not layers.

```cpp
#include "autograd/autograd.h"
#include "nn/layers/dropout.h"
#include "nn/layers/linear.h"
#include "nn/module.h"

class MLP : public nn::Module<float> {
public:
    nn::Linear<float> fc1{4, 8};
    nn::Linear<float> fc2{8, 2};
    nn::Dropout<float> drop{0.1};

    MLP() {
        register_module("fc1", fc1);
        register_module("fc2", fc2);
        register_module("drop", drop);
    }

    tensor::Tensor<float> forward(const tensor::Tensor<float>& x) {
        auto h = ops::relu(fc1.forward(x));
        h = drop.forward(h);               // non-const; respects train()/eval()
        return fc2.forward(h);
    }
};

MLP model;
model.train();                             // default anyway
nn::optim::Adam<float> opt(model.parameters(), 1e-3);

// inference: dropout becomes identity
model.eval();
{
    autograd::NoGradContext no_grad;
    auto out = model.forward(x);
}
```

`register_parameter("w", w)` is for a bare `Tensor` that is not inside a layer. Duplicate or empty names throw. See [layers/reference.md](layers/reference.md#module--layer).

---

## Convolution and pooling

NCHW layout. One `int64_t` kernel/stride/padding on every spatial axis. No `groups`.

```cpp
#include "autograd/autograd.h"
#include "nn/layers/convolutional.h"
#include "nn/layers/pooling.h"

nn::Conv2d<float> conv(1, 8, /*kernel=*/3, /*stride=*/1, /*padding=*/1);
nn::MaxPool2d<float> pool(2);              // stride defaults to kernel

auto x = tensor::Tensor<float>::randn({2, 1, 16, 16}, false, /*seed=*/1);
auto h = ops::relu(conv.forward(x));       // {2, 8, 16, 16}
auto y = pool.forward(h);                  // {2, 8, 8, 8}
```

Free functions (`ops::conv2d`, `ops::max_pool2d`, …) take an optional `bias` pointer and the same geometry kwargs. [ops.md](ops.md#convolution).

---

## Views, `narrow`, and `cat`

```cpp
auto t = tensor::Tensor<float>::arange(0, 12).reshape({3, 4});
auto row = t.narrow(/*dim=*/0, /*start=*/1, /*length=*/1);  // view of row 1
auto packed = t.transpose().contiguous();                   // gather if strided

auto a = tensor::Tensor<float>::ones({2, 3});
auto b = tensor::Tensor<float>::zeros({2, 3});
auto c = ops::cat({a, b}, /*dim=*/0);                       // {4, 3}
```

`narrow` is a view (`NarrowBackward` scatters the gradient back). `cat` always allocates. Binary elementwise ops do **not** broadcast; call `broadcast_to` first.

---

## Save and load

Build the destination model first. Load copies into existing tensors (`copy_`), so optimizer pointers stay valid.

```cpp
#include "nn/serialize.h"

nn::save(model, "model.agck");
MLP other;
nn::load(other, "model.agck");             // strict: names/shapes must match

nn::load(other, "model.agck", /*strict=*/false);  // skip missing/extra keys
```

Enable [logging](logging.md) to see save/load messages and non-strict warnings.

---

## Constraints that bite in practice

| Gotcha | What to do |
|---|---|
| `backward()` on a non-scalar | Reduce first (`sum`, a loss op) |
| Second backward through the same graph | Not supported (`retain_graph` does not exist). Rebuild the forward. |
| In-place `data()` / `[]` between forward and backward | Saved aliases see the mutated values; gradients are wrong |
| `view` on a transpose | Pack first: `t.contiguous().view(...)` or use `reshape` |
| Cross-entropy class indices | This library wants **soft labels** the same shape as logits |
| Empty optimizer param list | Throws. Pass `model.parameters()` after the layers exist |
