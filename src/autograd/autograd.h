#pragma once

// Layer 0: tensor data structure (no ops, no autograd graph)
#include "../tensor/tensor.h"

// Layer 1: autograd infrastructure
#include "grad_context.h"
#include "node.h"

// Layer 2: backward nodes (included transitively by the op headers below,
// but listed here for documentation of the dependency order)

// Layer 2.5: AccumulateGrad (leaf-tensor terminal node) and backward engine
#include "accumulate_grad.h"
#include "engine.h"

// Layer 3: forward ops (each header includes its own backward nodes)
#include "../ops/elementwise_ops.h"
#include "../ops/shape_ops.h"

// Layer 3.5: mixed-dtype overloads (must come after same-type ops are defined)
#include "../ops/mixed_dtype_ops.h"

// Layer 4: member-method wrappers on Tensor<T> (must come after all ops are defined)
#include "../tensor/tensor_methods.tpp"
