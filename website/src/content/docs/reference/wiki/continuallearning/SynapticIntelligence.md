---
title: "SynapticIntelligence<T, TInput, TOutput>"
description: "Synaptic Intelligence (SI) strategy for continual learning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ContinualLearning.Strategies`

Synaptic Intelligence (SI) strategy for continual learning.

## For Beginners

Synaptic Intelligence is similar to EWC but estimates weight
importance online during training rather than computing Fisher Information afterward.
It tracks how much each weight contributes to loss reduction using a "path integral".

## How It Works

**Key Insight:** SI measures how much each parameter contributed to learning,
not just how important it is for the current solution. This is done by integrating
the gradient signal along the learning trajectory.

**How it works:**

**The Math (Path Integral):**

ω_i = Σ_t (-∂L/∂θ_i(t)) × (θ_i(t) - θ_i(t-1))

This approximates the contribution of parameter i to loss reduction.

**Advantages over EWC:**

**Reference:** Zenke, F., Poole, B., and Ganguli, S.
"Continual Learning Through Synaptic Intelligence" (2017). ICML.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SynapticIntelligence(ILossFunction<>,)` | Initializes a new SI strategy with a lambda value. |
| `SynapticIntelligence(ILossFunction<>,SIOptions<>)` | Initializes a new SI strategy with custom options. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ConsolidatedImportance` | Gets a snapshot of the consolidated importance values. |
| `Damping` | Gets the damping constant. |
| `IsTrackingTask` | Gets whether the strategy is currently tracking a task. |
| `Lambda` | Gets the regularization strength. |
| `LayerImportance` | Gets per-layer importance statistics (if tracking enabled). |
| `MemoryUsageBytes` |  |
| `ModifiesArchitecture` |  |
| `Name` |  |
| `OptimalParameters` | Gets a snapshot of the optimal parameters from the last completed task. |
| `RequiresMemoryBuffer` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `AccumulateImportance(Vector<>)` | Accumulates task importance into the consolidated omega. |
| `AdjustGradients(Vector<>)` |  |
| `ComputeLayerStatistics(Vector<>)` | Computes approximate per-chunk importance statistics. |
| `ComputeMax(Vector<>)` | Computes the maximum value in a vector. |
| `ComputeMean(Vector<>)` | Computes the mean of a vector. |
| `ComputeRegularizationLoss(IFullModel<,,>)` |  |
| `CountNonZero(Vector<>)` | Counts non-zero elements in a vector. |
| `FinalizeTask(IFullModel<,,>)` |  |
| `GetStateForSerialization` |  |
| `HasPreviousTasks` | Checks if there are any previous tasks with non-zero importance. |
| `NormalizeImportanceValues(Vector<>)` | Normalizes importance values to prevent numerical issues. |
| `NotifyParameterUpdate(Vector<>)` | Notifies SI of a parameter update (call this after optimizer.step()). |
| `PrepareForTask(IFullModel<,,>,IDataset<,,>)` |  |
| `Reset` |  |
| `UpdatePathIntegral(Vector<>,Vector<>)` | Updates the path integral with current gradient and parameter change. |

