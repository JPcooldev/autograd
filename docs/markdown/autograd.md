# Autograd Engine

The autograd system implements **reverse-mode automatic differentiation** (backpropagation) using a dynamically built directed acyclic graph (DAG). The graph is constructed incrementally during the forward pass and consumed once during the backward pass.

Headers live under `src/autograd/`. Always include the library through the umbrella header:

```cpp
#include "autograd/autograd.h"
```

---

## Concepts

### Computation graph

Every time an operation runs on tensors that have `requires_grad = true`, the library:

1. Computes the forward result (the actual data).
2. Creates a concrete `Node<T>` subclass for that operation.
3. Wires the node's `next_edges` to the `grad_fn` (or `AccumulateGrad`) of each input.
4. Attaches the node to the **output** tensor's `grad_fn_`.

The result is a DAG where:

- **Output tensors** (results of ops) point to the node that created them (`grad_fn`).
- **Leaf tensors** (user-created, `requires_grad = true`) have `grad_fn = nullptr` and instead own an `AccumulateGrad` node that stores the gradient.
- **Edges** (`next_edges` inside a node) point toward the inputs that contributed to the output.

```
leaf x ─── AccumulateGrad ←─── AddBackward ←─── loss.grad_fn
leaf y ─── AccumulateGrad ←──/
```

### Leaves vs intermediates

| Property | Leaf tensor | Intermediate tensor |
|---|---|---|
| Created by | User code | An `ops::` function |
| `grad_fn` | `nullptr` | non-null node |
| `is_leaf()` | `true` | `false` |
| Receives gradient | yes (via `AccumulateGrad`) | passed through only |

### Gradient accumulation

Gradients are **accumulated** (added together) at each leaf node. This correctly handles the case where a leaf tensor is used multiple times in the computation graph — each path contributes its partial gradient.

---

## Gradient context

**Header:** `src/autograd/grad_context.h`

```cpp
bool autograd::is_grad_enabled();          // true by default (thread-local)
void autograd::set_grad_enabled(bool on);
```

### RAII guards

`AutoGradContext` and `NoGradContext` restore the previous state on destruction:

```cpp
{
    autograd::NoGradContext no_grad;
    // no nodes are created inside this scope
    auto y = ops::add(x, w);   // y.grad_fn == nullptr
}
// gradient tracking restored to previous state
```

```cpp
{
    autograd::set_grad_enabled(false);
    autograd::AutoGradContext restore_grad(true);
    // grad is enabled again inside this block
}
```

All `ops::` implementations check `is_grad_enabled()` before allocating a backward node, so disabling grad is zero-cost for the computation graph.

---

## `Node<T>`

**Header:** `src/autograd/node.h`

`Node<T>` is the abstract base class for all backward functions.

```cpp
template <typename T>
class Node {
public:
    virtual std::vector<Tensor<T>> apply(const Tensor<T>& propagated_grad) = 0;

    std::vector<std::shared_ptr<Node<T>>> next_edges;
    std::vector<Tensor<T>>               saved_tensors;
};
```

### `apply`

Called by the engine during the backward pass. Receives the gradient propagated from the output side and returns a vector of gradients for each input in the same order as `next_edges`.

### `next_edges`

Pointers to the backward nodes of the inputs. The engine follows these edges to propagate gradients deeper into the graph.

### `saved_tensors`

Deep copies of forward tensors that the backward formula needs (e.g. `MultiplyBackward` saves both `x` and `y`). Copies are made at node construction time via `snapshot_tensor`, which creates a plain owning tensor with `requires_grad = false` to avoid reference cycles.

### `get_next_edge` (static)

```cpp
static std::shared_ptr<Node<T>> get_next_edge(const Tensor<T>& x);
```

Returns the correct backward edge for a tensor:

- If `x` is an intermediate tensor → `x.grad_fn()`
- If `x` is a leaf with `requires_grad` → `x.ensure_accumulate_grad_fn()` (lazily created)
- Otherwise → `nullptr`

This is called inside every `ops::` implementation when wiring a new backward node.

---

## `AccumulateGrad<T>`

**Header:** `src/autograd/accumulate_grad.h`

A terminal node attached to leaf tensors. It has no `next_edges`. Its `apply` either copies the incoming gradient (first call) or accumulates it (`+=`).

```cpp
// AccumulateGrad stores a shared_ptr<Tensor<T>> internally.
// Access via Tensor::grad():
auto g = x.grad();   // returns shared_ptr<Tensor<T>>, null if not yet computed
```

---

## Backward engine

**Header:** `src/autograd/engine.h`

```cpp
template <typename T>
void autograd::run_backward(
    const std::shared_ptr<Node<T>>& root_fn,
    const tensor::Tensor<T>&        initial_grad
);
```

### Algorithm

1. **Topological sort** — DFS post-order from `root_fn`, then reversed, so the root is processed first and leaves last.

2. **Seed accumulator** — initialises the gradient accumulator map with `root_fn → initial_grad`.

3. **Backward loop** — for each node in topological order:
   - Look up the accumulated gradient for this node.
   - Call `node->apply(accumulated_grad)` → get per-input gradients.
   - For each `next_edge[i]`: add `input_grads[i]` into that edge's accumulator (creating it if this is the first contribution).

4. **Leaf accumulation** — when the loop reaches an `AccumulateGrad` node, its `apply` stores or adds the gradient into the leaf tensor's `.grad()`.

### Calling `backward`

`Tensor::backward()` is the user-facing entry point:

```cpp
loss.backward();
```

It:
1. Validates `numel() == 1` (scalar output required).
2. Validates `grad_fn_` is non-null (must be a non-leaf tensor that resulted from an op).
3. Constructs the scalar seed `Tensor<T>({}, {T{1}}, false)`.
4. Calls `autograd::run_backward(grad_fn_, seed)`.

There is no support for:
- Non-scalar loss (custom `gradient` argument).
- Retaining the graph for multiple backward calls through the same graph.

---

## Backward node implementations

### Elementwise

**Header:** `src/autograd/backward_ops/backward_elementwise_ops.h`

| Node | Formula |
|---|---|
| `AddBackward` | `grad_x = grad`, `grad_y = grad` |
| `SubtractBackward` | `grad_x = grad`, `grad_y = -grad` |
| `MultiplyBackward` | `grad_x = saved_y * grad`, `grad_y = saved_x * grad` |
| `DivideBackward` | `grad_x = (1/saved_y) * grad`, `grad_y = (-saved_x / saved_y²) * grad` |
| `NegationBackward` | `grad_x = -grad` |
| `PowerBackward` | `grad_x = exponent * saved_x^(exponent-1) * grad` |
| `AbsBackward` | `grad_x = sign(saved_x) * grad` |
| `ExpBackward` | `grad_x = exp(saved_x) * grad` |
| `LogBackward` | `grad_x = (1 / saved_x) * grad` |
| `SinBackward` | `grad_x = cos(saved_x) * grad` |
| `CosBackward` | `grad_x = -sin(saved_x) * grad` |
| `TanBackward` | `grad_x = (1 / cos²(saved_x)) * grad` |
| `SigmoidBackward` | `grad_x = σ(saved_x) * (1 − σ(saved_x)) * grad` |
| `ReluBackward` | `grad_x = (saved_x > 0) * grad` |

### Shape

**Header:** `src/autograd/backward_ops/backward_shape_ops.h`

| Node | Backward |
|---|---|
| `TransposeBackward` | `grad.transpose(dim0, dim1)` |
| `ReshapeBackward` | `grad.reshape(original_shape)` |
| `BroadcastToBackward` | `sum_to(grad, original_shape)` — sums along broadcast axes |
| `FlattenBackward` | `grad.reshape(original_shape)` |
| `SqueezeBackward` | `grad.reshape(original_shape)` |
| `UnsqueezeBackward` | `grad.squeeze(dim)` |
| `CastBackward<Out, In>` | Recast `grad` from `Out` to `In`. Same-type is identity; cross-type bridges into the input graph. |

`sum_to` is an internal helper that reduces a gradient tensor to a target shape by summing along all dimensions that were broadcast or added.

### Reduction

**Header:** `src/autograd/backward_ops/backward_reduction_ops.h`

| Node | Backward |
|---|---|
| `SumBackward` | `broadcast_to(grad, original_shape)` |
| `SumDimBackward` | `unsqueeze(grad, dim)` then `broadcast_to(original_shape)` |
| `MeanBackward` | `broadcast_to(grad / numel, original_shape)` |
| `MeanDimBackward` | Same as `SumDimBackward` scaled by `1/size(dim)` |
| `MaxBackward` | One-hot at saved `argmax` position |
| `MinBackward` | One-hot at saved `argmin` position |

### Linalg

**Header:** `src/autograd/backward_ops/backward_linalg_ops.h`

| Node | Backward |
|---|---|
| `DotBackward` | `grad_a = b * scalar_grad`, `grad_b = a * scalar_grad` |
| `MatmulBackward` | `grad_A = grad_C @ B^T`, `grad_B = A^T @ grad_C` |

`MatmulBackward` stores contiguous copies of `A` and `B` at construction time (via an internal `to_contiguous_2d` helper) to ensure the triple-loop backward computation has a guaranteed memory layout.

---

## Adding a new operation

Every new differentiable operation follows the same three-step pattern:

### 1. Forward kernel (`ops/`)

```cpp
// src/ops/my_ops.h
template <typename T>
tensor::Tensor<T> my_op(const tensor::Tensor<T>& x) {
    std::vector<T> storage(x.numel());
    for (size_t i = 0; i < storage.size(); ++i)
        storage[i] = /* forward formula */;

    const bool rg = autograd::is_grad_enabled() && x.requires_grad();
    std::shared_ptr<autograd::Node<T>> grad_fn;
    if (rg)
        grad_fn = std::make_shared<autograd::MyOpBackward<T>>(x);

    return tensor::Tensor<T>::from_operation_result(
        x.shape(), std::move(storage), rg, std::move(grad_fn));
}
```

### 2. Backward node (`autograd/backward_ops/`)

```cpp
// src/autograd/backward_ops/backward_my_ops.h
template <typename T>
class MyOpBackward : public Node<T> {
public:
    explicit MyOpBackward(const tensor::Tensor<T>& x) : Node<T>(x) {}

    std::vector<tensor::Tensor<T>> apply(
        const tensor::Tensor<T>& grad) override
    {
        const auto& saved_x = this->saved_tensors[0];
        // compute grad_x using saved_x and incoming grad
        auto grad_x = /* backward formula */;
        return {grad_x};
    }
};
```

### 3. Wiring in `Tensor` (optional)

Add a method in `tensor.h` (declaration) and `tensor_methods.tpp` (definition) that delegates to the free function:

```cpp
// tensor.h
Tensor<T> my_op() const;

// tensor_methods.tpp
template <typename T>
Tensor<T> Tensor<T>::my_op() const {
    return ops::my_op(*this);
}
```

Include the new headers in `autograd/autograd.h` in the correct order (ops headers before `tensor_methods.tpp`).

---

## Thread safety

The gradient-enabled flag is **thread-local**, so each thread has independent control. However, the tensors themselves and their shared data buffers are **not** thread-safe for concurrent writes. Do not share a `Tensor` (or its leaf gradient) across threads without external synchronization.
