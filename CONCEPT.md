# EduTorch: Concepts & Design of a Minimal Educational C++ Tensor + Autograd Library

**Version:** 1.0 (March 2026)  
**Author:** Your educational from-scratch PyTorch  
**Goal:** Learn the real internals of tensors and automatic differentiation without the 500k-line complexity of real PyTorch.

---

## 1. Why We Are Building This

Real PyTorch is huge because it supports:

- CPU + CUDA + XLA + MPS + custom accelerators
- 30+ dtypes
- Distributed training, TorchScript, TorchDynamo, etc.

Our **EduTorch** has one purpose only:

- Teach the **core ideas** that make PyTorch work.
- Stay under ~2000 lines of clean C++.
- Use a **templated `Tensor<T>`** approach (float, double, int, …) so everything is type-safe and simple.

We deliberately **do not** copy PyTorch’s exact code. We copy its **concepts** in the cleanest possible way.

---

## 2. The Tensor Memory Model – The Foundation

### 2.1 Two Ways to Represent Memory

**Old PyTorch way (2019):** Tensor + separate Storage  
**Our way (NumPy-style, recommended in ezyang’s blog):** Single Tensor class + `base_` pointer

**Why we chose the base-tensor model:**

- Owning tensors have **zero indirection** (data lives directly inside the object).
- Views are just metadata + a pointer to the base tensor.
- One refcount instead of two.
- Better view chaining.
- Matches exactly how NumPy works.

### 2.2 Internal State of Every Tensor

Every `Tensor` object privately holds:

```text  
std::shared_ptr> data_buffer_;  // only non-null in owning tensor  
Tensor* base_;                                      // null for owning tensors  
size_t offset_;                                        // byte offset into the base data  
std::vector sizes_;                           // shape  
std::vector strides_;                         // stride per dimension  
Dtype dtype_;                                          // stored for runtime checks  
bool requires_grad_ = false;  
std::shared_ptr> grad_fn_;                     // ← the autograd hook
```

**Why `shared_ptr<vector<byte>>`?**

- Only the **owning** tensor allocates the buffer.
- Every view simply copies the `shared_ptr` → automatic reference counting.
- Memory stays alive as long as **any** tensor in the view tree exists.
- No manual `delete` or raw pointers → safe and educational.

---

## 3. Views, Strides, Contiguity & Broadcasting

### 3.1 Views Are Just Metadata

`view()`, `reshape()`, `transpose()`, `narrow()`, `squeeze()`, `expand()` **never** copy data.

They create a **new** `Tensor<T>` that:

- Points to the same `data_buffer`_ (via `shared_ptr`)
- Has its own `sizes_`, `strides_`, `offset_`
- Sets `base_` to the original tensor (or chains through multiple views)

**Contiguity check** (very important):

```text
bool is_contiguous() const;
```

A tensor is contiguous when `strides[i] == product_of_sizes_after_i`.

### 3.2 Strides – The Magic of Views

Strides tell us how to jump in memory for any index.

Example: 2×3 tensor with stride (3,1) → normal row-major  
Slice of first column → stride becomes (3,3) → we jump 3 elements each time.

All indexing, slicing, and math ops use the stride formula:

```text
index = offset + sum( idx[i] * stride[i] )
```

### 3.3 Broadcasting

Broadcasting is **not** copying data — it is just **pretending** a tensor has a larger shape by using stride = 0 in those dimensions.

We need a small helper:

```text
std::pair<std::vector<int64_t>, std::vector<std::vector<int64_t>>> broadcast_shapes(...)
```

Used by every binary op and reduction.

---

## 4. Dtype Handling – Why Templates?

We made `Tensor<T>` templated (`Tensor<float>`, `Tensor<double>`, `Tensor<int32_t>`, …).

**Advantages for education:**

- Zero runtime cost (everything is compile-time).
- Type-safe math (you can’t accidentally add float and int without explicit cast).
- No giant `switch` on every operation.
- Compiler generates specialized code for each dtype automatically.

**Limitation we accept:** All tensors in one computation graph must have the **same** `T`.  
(This is fine for learning — real models almost always use one dtype anyway.)

If you later want mixed dtypes, you would need a non-templated base class (much harder).

---

## 5. Automatic Differentiation – The Big Picture

Autograd = **reverse-mode automatic differentiation** using a **Dynamic Acyclic Graph (DAG)**.

Key insight:

> We never build the backward graph in advance.  
> We **extend** it automatically every time a forward operation runs.

The entire graph is built from two things:

1. `Tensor<T>::grad_fn_` → pointer to the Node that created this tensor
2. `Node<T>` objects that point to their **parents** (`next_edges`)

---

## 6. The Node Class – Heart of the Autograd Engine

```text
template<typename T>
class Node {
public:
    virtual std::vector<Tensor<T>> apply(const std::vector<Tensor<T>>& grad_outputs) = 0;

    std::vector<std::shared_ptr<Node<T>>> next_edges;   // "children" = input tensors' grad_fn
    std::vector<Tensor<T>> saved_tensors;               // values needed for backward

    virtual ~Node() = default;
};
```

**What each part does:**

- `apply(grad_outputs)` = the **local derivative** (the mathematical backward function).
- `next_edges` = links back to the tensors that were inputs to the forward op.
- `saved_tensors` = copies of forward values that the backward formula needs (e.g. original `x` and `y` for multiplication).

---

## 7. How Operations Add Nodes to the Graph

Every forward operation follows **exactly** the same pattern:

1. Compute the actual data using a pure kernel (`ops::add`, `ops::mul`, …).
2. If `requires_grad` is true on any input:
  - Create a concrete subclass of `Node<T>` (e.g. `AddBackward<T>`, `MulBackward<T>`).
  - Wire `next_edges` to the input tensors’ `grad_fn`.
  - Save whatever is needed into `saved_tensors`.
  - Attach this **new node** to the **output tensor**’s `grad_fn_`.
3. Return the output tensor.

**Example for addition:**

```text
Tensor<T> result = ops::add(a, b);

if (a.requires_grad() || b.requires_grad()) {
    auto node = std::make_shared<AddBackward<T>>();
    node->next_edges = {a.grad_fn_, b.grad_fn_};
    // add needs no saved tensors
    result.grad_fn_ = std::move(node);
}
```

This is the **only** place where the graph grows.

---

## 8. Concrete Backward Nodes (Examples)

### AddBackward

```text
grad_x = grad_z
grad_y = grad_z
```

### MulBackward

```text
grad_x = grad_z * y
grad_y = grad_z * x
```

### View/Transpose Backward

Usually an **IdentityBackward** or no new node at all (gradient flows straight through).

### SumBackward

Needs to know the original shape to **expand** the incoming gradient back to the input shape.

Every new operation you add requires:

- A forward kernel in `ops::`
- A `XXXBackward<T>` class
- The wiring code inside the Tensor method

---

## 9. The Backward Pass (Traversal)

Two possible implementations:

### Simple recursive version (great for learning)

```text
void Tensor<T>::backward() {
    if (!grad_fn_) {
        grad_ = ones_like(*this);   // leaf tensor
        return;
    }

    auto grads_in = grad_fn_->apply({grad_});
    for (size_t i = 0; i < grads_in.size(); ++i) {
        // find the i-th input tensor
        input_tensor.grad_ += grads_in[i];
        input_tensor.backward();   // recurse
    }
}
```

### Production version (what real PyTorch uses)

- A **ready queue** (topological order)
- Reference counting on Nodes so we know when all downstream gradients have arrived
- Handles multiple outputs, in-place ops, etc.

For EduTorch, start with the recursive version, then upgrade to the queue when you want to support deeper graphs.

---

## 10. Gradient Accumulation & Leaves

- Leaf tensors (parameters, inputs) have `grad_fn_ = nullptr`.
- When gradient reaches them, we create a special `AccumulateGrad<T>` node (or just add directly to `tensor.grad_`).
- Gradients are **accumulated** (`+=`) because a tensor can be used multiple times in the forward pass.

---

## 11. Recommended Implementation Roadmap

**Phase 0** – Tensor core

- Internal state, constructors, destructor
- Views (view, transpose, reshape, narrow, squeeze, expand)
- Strides, contiguity, indexing, broadcasting helper

**Phase 1** – Math foundation

- Element-wise: add, sub, mul, div, neg, abs
- Reductions: sum, mean, max, min
- Matrix: matmul, mm
- In-place versions

**Phase 2** – Autograd

- Add `grad_fn_` and `Node<T>` base
- Implement AddBackward, MulBackward, SumBackward, IdentityBackward
- Wire nodes in every forward op
- Implement `Tensor::backward()`

**Phase 3** – Polish

- `requires_grad_`, `detach()`, `zero_grad()`
- SavedVariable to avoid cycles
- In-place version counter
- Random, fill, comparison ops

---

## 12. Why This Design Is Clean and Educational

- No giant code generation
- No separate Storage class
- Everything is templated → no runtime dtype dispatch
- Graph is built **incrementally** during forward pass (exactly like real PyTorch)
- Backward functions are pure mathematics (`apply()` = local chain rule)
- You can read the entire autograd engine in < 300 lines

This is the same mental model Edward Yang explained in his 2019 “PyTorch Internals” talk — just simplified for learning.

---

## 13. Common Pitfalls & Gotchas

1. **Reference cycles** → use `std::weak_ptr` or `SavedVariable` for saved_tensors.
2. **In-place operations** → need version counter to detect mutation during backward.
3. **Broadcasting in backward** → must expand gradients correctly.
4. **Views + autograd** → gradient must flow back to the original base tensor.
5. **Scalar vs tensor gradients** → `loss.backward()` implicitly starts with `grad = 1.0`.

---

## Final Words

You now have the complete conceptual map.

Start coding **Phase 0** (Tensor memory + views) first.  
Once that works, adding autograd becomes surprisingly mechanical — every new operation is just “kernel + Node subclass + wiring”.

This design is small enough to understand in a weekend, yet deep enough that you will truly understand how real PyTorch works under the hood.

Happy coding!  
When you finish Phase 1, come back and we’ll implement the autograd engine together.

**References**  

- ezyang’s “PyTorch Internals” blog (2019)  
- NumPy memory model  
- Original Torch7 / Chainer papers





---
# Ops

## elementwise operations
- add
    - z = x + y
    - grad_x = 1 * grad_z
    - grad_y = 1 * grad_z
- negation
    - z = -x
    - grad_x = -1 * grad_z
- subtract
    - z = x - y = x + (-1) * y
    - grad_x = 1 * grad_z
    - grad_y = -1 * grad_z
- multiply
    - z = x * y
    - grad_x = y * grad_z
    - grad_y = x * grad_z
- divide
    - z = x / y
    - grad_x = 1 / y * grad_z
    - grad_y = -x / y^2 * grad_z
- power
    - z = x ^ constant
    - grad_x = constant * x ^ (constant - 1) * grad_z
    - grad_y = 0
- abs
    - z = abs(x)
    - grad_x = sign(x) * grad_z, where sign(x) is 1 if x > 0, 0 if x == 0, -1 if x < 0
- exp
    - z = exp(x)
    - grad_x = exp(x) * grad_z
- log
    - z = log(x)
    - grad_x = 1 / x * grad_z
- sin
    - z = sin(x)
    - grad_x = cos(x) * grad_z
- cos
    - z = cos(x)
    - grad_x = -sin(x) * grad_z
- tan
    - z = tan(x)
    - grad_x = 1 / cos^2(x) * grad_z
