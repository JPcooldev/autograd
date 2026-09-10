# Optimizer reference

C++ constructors for `nn::optim`. Update formulas: [theory.md](theory.md). How `step()` reads `.grad()`: [mechanism.md](mechanism.md).

```cpp
#include "nn/optim/sgd.h"
#include "nn/optim/adam.h"
#include "nn/optim/adamw.h"
```

All three take a **non-empty** `std::vector<tensor::Tensor<T>*>` (typically `model.parameters()`). An empty list throws `std::invalid_argument`. The tensors must outlive the optimizer.

```cpp
nn::Linear<float> model(4, 2);
nn::optim::SGD<float>  sgd(model.parameters(), 0.01);
nn::optim::Adam<float> adam(model.parameters(), 1e-3, 0.9, 0.999, 1e-8, 1e-4);
nn::optim::AdamW<float> adamw(model.parameters(), 1e-3);  // weight_decay defaults to 0.01
```

`step()` writes `param->data()` in place. `zero_grad()` clears gradient buffers only — Adam `m` / `v` / `step` stay.

A parameter is skipped when `!requires_grad()` or `grad() == nullptr`. Adam / AdamW **do not increment that parameter’s step** in that case.

There is no momentum SGD, no param groups, and no public getters for `learning_rate_`.

---

## `Optimizer<T>`

**Header:** `src/nn/optim/optimizer.h`

Base class. Not instantiated directly.

```cpp
Optimizer(parameters, learning_rate, weight_decay=0.0)
virtual void step() = 0
void zero_grad()
```

---

## SGD

**Header:** `src/nn/optim/sgd.h`

Coupled L2: \(\theta \leftarrow \theta - \eta(g + \lambda\theta)\). No extra state.

```cpp
nn::optim::SGD<T>(
    std::vector<tensor::Tensor<T>*> parameters,
    double learning_rate,
    double weight_decay = 0.0
)
```

| Argument | Default | Role |
|---|---|---|
| `parameters` | (required) | Leaf tensors to update |
| `learning_rate` | (required) | \(\eta\) |
| `weight_decay` | `0` | \(\lambda\), added into the gradient |

---

## Adam

**Header:** `src/nn/optim/adam.h`

Coupled decay **inside** the gradient before the moments (`g_eff = g + λθ`). Defaults match PyTorch Adam.

```cpp
nn::optim::Adam<T>(
    std::vector<tensor::Tensor<T>*> parameters,
    double learning_rate,
    double beta1 = 0.9,
    double beta2 = 0.999,
    double eps = 1e-8,
    double weight_decay = 0.0
)
```

Per-parameter state (`AdamState`): first moment `m`, second moment `v`, integer `step`. Sized from `param->data().size()` in the constructor.

---

## AdamW

**Header:** `src/nn/optim/adamw.h`

Moments on raw `g`. Decoupled decay: \(\theta \leftarrow \theta - \eta\,\hat{m}/(\sqrt{\hat{v}}+\varepsilon) - \eta\lambda\theta\). Default \(\lambda = 0.01\) (PyTorch AdamW).

```cpp
nn::optim::AdamW<T>(
    std::vector<tensor::Tensor<T>*> parameters,
    double learning_rate,
    double beta1 = 0.9,
    double beta2 = 0.999,
    double eps = 1e-8,
    double weight_decay = 0.01
)
```
