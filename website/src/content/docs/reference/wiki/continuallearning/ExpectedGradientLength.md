---
title: "ExpectedGradientLength<T, TInput, TOutput>"
description: "Expected Gradient Length (EGL) strategy for continual learning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ContinualLearning.Strategies`

Expected Gradient Length (EGL) strategy for continual learning.

## For Beginners

EGL protects important parameters by measuring how much
each parameter affects the output when training on a task. Parameters with larger
expected gradient lengths are considered more important.

## How It Works

**How it works:**

**The Math:**

EGL loss: L_total = L_task + (λ/2) * Σᵢ Ωᵢ * (θᵢ - θ*ᵢ)²

Where Ωᵢ = E[|∂L/∂θᵢ|] is the expected gradient length for parameter i

**Comparison to Other Methods:**

EGL is simpler than EWC and provides a direct measure of how much each
parameter contributes to the loss during training.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ExpectedGradientLength(ILossFunction<>,,Int32)` | Initializes a new EGL strategy with default options. |
| `ExpectedGradientLength(ILossFunction<>,EGLOptions<>)` | Initializes a new EGL strategy with custom options. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Importance` | Gets the accumulated importance weights. |
| `Lambda` | Gets the regularization strength (lambda). |
| `MemoryUsageBytes` |  |
| `ModifiesArchitecture` |  |
| `Name` |  |
| `PreviousParameters` | Gets the stored parameters from previous tasks. |
| `RequiresMemoryBuffer` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `AdjustGradients(Vector<>)` |  |
| `ComputeMax(Vector<>)` | Computes the maximum value in a vector. |
| `ComputeMean(Vector<>)` | Computes the mean of a vector. |
| `ComputeRegularizationLoss(IFullModel<,,>)` |  |
| `ComputeTaskImportance` | Computes the importance for the current task from accumulated gradient lengths. |
| `FinalizeTask(IFullModel<,,>)` |  |
| `GetStateForSerialization` |  |
| `NormalizeImportanceVector(Vector<>)` | Normalizes importance values to [0, 1] range. |
| `PrepareForTask(IFullModel<,,>,IDataset<,,>)` |  |
| `Reset` |  |
| `UpdateAccumulatedImportance(Vector<>,Vector<>)` | Updates accumulated importance with new task importance using exponential decay. |

