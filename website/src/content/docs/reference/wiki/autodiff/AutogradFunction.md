---
title: "AutogradFunction<T>"
description: "Base class for custom autograd functions with user-defined forward and backward passes."
section: "API Reference"
---

`Base Classes` · `AiDotNet.Autodiff`

Base class for custom autograd functions with user-defined forward and backward passes.
Equivalent to PyTorch's `torch.autograd.Function`.

## How It Works

Subclass this to define operations with custom gradient computation. The forward method
performs the computation and saves any tensors needed for backward. The backward method
receives the output gradient and returns input gradients.

**Usage:**

## Methods

| Method | Summary |
|:-----|:--------|
| `Apply(Tensor<>[])` | Applies this custom function: runs forward and registers backward on the active tape. |
| `Backward(AutogradContext,Tensor<>)` | Computes gradients w.r.t. |
| `Forward(AutogradContext,Tensor<>[])` | Performs the forward computation. |

