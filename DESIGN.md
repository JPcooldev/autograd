# Notes for design decisions

## 1. In-place ops
- do not allow in-place ops on the tensor since it will mutate the underlying data and break the computation graph
- yeah but a user can do tensor[i] = 7; in-place op that would mutate the data. how to do version check like PyTorch does? then i would throw an error if the version do not match in graph
- btw intermediate results are never mutated in-place (user does not have access to them)

## Data retention in the computation graph
- user owned tensor are never removed
- intermediate results might go out of local scope (e.g. end of forward pass function), but we keep their alias in the computation graph so we can still use them for backward pass (this reference will keep it alive)

## Aliasing tensors
- increase ref count on shared pointers (data_ptr_, base_ptr_, grad_fn_ and accumulate_grad_fn_)
- deep copy metadata
- decrease ref count by setting it to nullptr (disconnect from computation graph)
- disconnecting them from the computation graph is because Node uses alias to save reference to the input tensors (saved_tensors). this would create double references and break the computation graph

## Views
- views should share required_grad flag and grad buffer
- example is using `[CLS]` token and transforming it through prediction head. this should be part of the computational graph. we disconnect the view from the computation graph, but other math operation on this view should be reflected in the computation graph