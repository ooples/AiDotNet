---
title: "ISecondOrderGradientComputable<T, TInput, TOutput>"
description: "Extended gradient computation interface for MAML meta-learning algorithms."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Extended gradient computation interface for MAML meta-learning algorithms.

## For Beginners

While basic gradient computation tells you how to improve on a single task, second-order
gradients tell you how changing your starting point would affect learning on that task.

Think of it like this:

- First-order: "If I start here, which direction improves this task?"
- Second-order: "If I started slightly differently, how would my entire learning trajectory change?"

This is computationally expensive but more accurate for meta-learning.

## How It Works

This interface extends `IGradientComputable` with second-order
gradient computation capability required for full MAML (Model-Agnostic Meta-Learning).

**MAML Use Case:**
Full MAML uses second-order gradients to backpropagate through the inner loop adaptation,
computing true meta-gradients that account for how the adaptation process itself changes.
Reptile and first-order MAML approximate this with only first-order gradients.

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeSecondOrderGradients(List<ValueTuple<,>>,,,ILossFunction<>,)` | Computes second-order gradients (Hessian-vector product) for full MAML meta-learning. |

