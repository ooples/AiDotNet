---
title: "IDetachedTensor<T>"
description: "Marker interface for tensors that should not receive gradients."
section: "API Reference"
---

`Interfaces` · `AiDotNet.SelfSupervisedLearning`

Marker interface for tensors that should not receive gradients.

## How It Works

This can be used to mark tensors at compile time as non-differentiable.
Useful for type-safe gradient handling in advanced scenarios.

## Properties

| Property | Summary |
|:-----|:--------|
| `Data` | Gets the underlying tensor data. |

