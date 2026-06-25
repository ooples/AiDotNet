---
title: "LossFunctionBase<T>"
description: "Base class for loss function implementations."
section: "API Reference"
---

`Base Classes` · `AiDotNet.LossFunctions`

Base class for loss function implementations.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LossFunctionBase` | Initializes a new instance of the LossFunctionBase class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Engine` | Gets the global execution engine for vector operations. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateDerivative(Vector<>,Vector<>)` | Calculates the derivative (gradient) of the loss function. |
| `CalculateLoss(Vector<>,Vector<>)` | Calculates the loss between predicted and actual values. |
| `CalculateLossAndGradientGpu(Tensor<>,Tensor<>)` | Calculates both loss and gradient on GPU in a single pass. |
| `ComputeTapeLoss(Tensor<>,Tensor<>)` | Computes the loss as a scalar tensor using tape-differentiable engine operations. |
| `EnsureTargetMatchesPredicted(Tensor<>,Tensor<>)` | When the target has fewer dimensions than the prediction (e.g., integer class indices `[B, S]` vs logits `[B, S, V]`), auto-converts to one-hot encoding so pointwise loss operations (multiply, subtract) work without shape mismatch. |
| `ValidateVectorLengths(Vector<>,Vector<>)` | Validates that the predicted and actual vectors have the same length. |

## Fields

| Field | Summary |
|:-----|:--------|
| `NumOps` | Provides operations for the numeric type T. |

