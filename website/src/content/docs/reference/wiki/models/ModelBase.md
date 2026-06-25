---
title: "ModelBase<T, TInput, TOutput>"
description: "Abstract base class for standalone models that directly implement `IFullModel`."
section: "API Reference"
---

`Base Classes` · `AiDotNet.Models`

Abstract base class for standalone models that directly implement `IFullModel`.

## For Beginners

This is the foundation for building standalone machine learning models.
Models like linear regression, expression trees, gradient boosting, and ensembles all inherit
from this class. It handles boilerplate like serialization and feature tracking so each model
only needs to implement its core prediction and training logic.

## How It Works

Provides common infrastructure and sensible defaults for standalone model implementations
that are not wrappers around other models. Subclasses must implement core model behavior:
prediction, training, parameter management, loss function, and cloning.

## Properties

| Property | Summary |
|:-----|:--------|
| `DefaultLossFunction` |  |
| `Engine` | Gets the hardware-accelerated computation engine for vectorized operations. |
| `ParameterCount` |  |
| `SupportsParameterInitialization` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyGradients(Vector<>,)` |  |
| `Clone` |  |
| `ComputeGradients(,,ILossFunction<>)` |  |
| `DeepCopy` |  |
| `Deserialize(Byte[])` |  |
| `Dispose` |  |
| `Dispose(Boolean)` | Releases resources held by this model. |
| `GetActiveFeatureIndices` |  |
| `GetFeatureImportance` |  |
| `GetModelMetadata` |  |
| `GetParameterChunks` |  |
| `GetParameters` |  |
| `IsFeatureUsed(Int32)` |  |
| `LoadModel(String)` |  |
| `LoadState(Stream)` |  |
| `Predict()` |  |
| `SanitizeParameters(Vector<>)` |  |
| `SaveModel(String)` |  |
| `SaveState(Stream)` |  |
| `Serialize` |  |
| `SetActiveFeatureIndices(IEnumerable<Int32>)` |  |
| `SetParameters(Vector<>)` |  |
| `Train(,)` |  |
| `WithParameters(Vector<>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `NumOps` | Numeric operations for type T. |

