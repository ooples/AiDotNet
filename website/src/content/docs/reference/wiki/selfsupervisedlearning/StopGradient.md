---
title: "StopGradient<T>"
description: "Provides stop-gradient operations for self-supervised learning."
section: "API Reference"
---

`Helpers & Utilities` · `AiDotNet.SelfSupervisedLearning`

Provides stop-gradient operations for self-supervised learning.

## For Beginners

Stop-gradient (also called "detach" in PyTorch) prevents gradients
from flowing through a tensor during backpropagation. This is crucial for several SSL methods:

## How It Works

**Why stop-gradient?**

Without stop-gradient, the model could "cheat" by making both branches output constants,
resulting in representation collapse. Stop-gradient forces asymmetry that prevents this.

**Example usage:**

## Methods

| Method | Summary |
|:-----|:--------|
| `Detach(Tensor<>)` | Detaches a tensor from the computation graph, preventing gradient flow. |
| `Detach(Vector<>)` | Applies stop-gradient to a vector. |
| `DetachBatch(Tensor<>[])` | Detaches a batch of tensors from the computation graph. |
| `SymmetricLoss(Tensor<>,Tensor<>,Tensor<>,Tensor<>,Func<Tensor<>,Tensor<>,>)` | Computes the symmetric loss with stop-gradient for SimSiam-style training. |
| `ZeroGrad(Tensor<>)` | Creates a zero-gradient version of a tensor for the backward pass. |

