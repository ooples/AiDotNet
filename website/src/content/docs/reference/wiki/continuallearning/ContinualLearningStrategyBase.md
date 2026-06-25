---
title: "ContinualLearningStrategyBase<T, TInput, TOutput>"
description: "Abstract base class for continual learning strategies providing common functionality."
section: "API Reference"
---

`Base Classes` · `AiDotNet.ContinualLearning.Strategies`

Abstract base class for continual learning strategies providing common functionality.

## For Beginners

This base class provides common functionality that all
continual learning strategies share, such as:

## How It Works

Derived classes implement specific algorithms like EWC, LwF, GEM, SI, and MAS.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ContinualLearningStrategyBase(ILossFunction<>)` | Initializes a new instance of the strategy base. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Engine` | Gets the global execution engine for hardware-accelerated vector operations. |
| `MemoryUsageBytes` |  |
| `ModifiesArchitecture` |  |
| `Name` |  |
| `RequiresMemoryBuffer` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `AdjustGradients(Vector<>)` |  |
| `CloneVector(Vector<>)` | Clones a vector. |
| `ComputeDotProduct(Vector<>,Vector<>)` | Computes the dot product of two vectors. |
| `ComputeL2Norm(Vector<>)` | Computes the L2 norm of a vector. |
| `ComputeRegularizationLoss(IFullModel<,,>)` |  |
| `EstimateVectorMemory(Vector<>)` | Estimates the memory usage of a Vector in bytes. |
| `FinalizeTask(IFullModel<,,>)` |  |
| `GetIncompatibilityReason(IFullModel<,,>)` |  |
| `GetMetrics` |  |
| `GetStateForSerialization` | Gets the state for serialization. |
| `GetTypeSize` | Gets the size of type T in bytes. |
| `IsCompatibleWith(IFullModel<,,>)` |  |
| `Load(String)` |  |
| `LoadStateFromSerialization(Dictionary<String,JsonElement>)` | Loads state from serialized data. |
| `PrepareForTask(IFullModel<,,>,IDataset<,,>)` |  |
| `RecordMetric(String,Object)` | Records a metric value. |
| `Reset` |  |
| `Save(String)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `CreatedAt` | Timestamp when the strategy was created. |
| `LossFunction` | The loss function used for computing task losses. |
| `Metrics` | Metrics collected during training. |
| `NumOps` | Numeric operations for the generic type T. |
| `TaskCount` | Number of tasks that have been processed. |

