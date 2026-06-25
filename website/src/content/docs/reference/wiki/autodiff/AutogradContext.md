---
title: "AutogradContext"
description: "Context object passed to `AutogradFunction` for saving tensors between forward and backward."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Autodiff`

Context object passed to `AutogradFunction` for saving tensors between forward and backward.
Equivalent to PyTorch's `ctx` in autograd.Function.

## Properties

| Property | Summary |
|:-----|:--------|
| `SavedCount` | Gets the number of saved tensors. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetSaved(Int32)` | Retrieves a saved tensor by index. |
| `SaveForBackward(Tensor<>)` | Saves a tensor for use in the backward pass. |
| `SaveForBackward(Tensor<>[])` | Saves multiple tensors for use in the backward pass. |

