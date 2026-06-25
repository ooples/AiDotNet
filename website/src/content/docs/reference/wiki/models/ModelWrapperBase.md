---
title: "ModelWrapperBase<T, TInput, TOutput>"
description: "Abstract base class for model wrappers that delegate to an underlying `IFullModel`."
section: "API Reference"
---

`Base Classes` · `AiDotNet.Models`

Abstract base class for model wrappers that delegate to an underlying `IFullModel`.

## For Beginners

Some models work by wrapping another model and adding extra behavior.
For example, a transfer-learning model wraps a pre-trained model with a feature mapper,
or an adversarial defense wraps a model with input preprocessing. This base class handles
all the common delegation so wrapper classes only implement what's different.

## How It Works

Provides default implementations for most `IFullModel`
interface members by delegating to the wrapped base model. Subclasses only need to override
prediction logic and parameter management specific to their wrapping strategy.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ModelWrapperBase(IFullModel<,,>)` | Initializes a new instance of the `ModelWrapperBase` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `BaseModel` | The underlying full model being wrapped. |
| `DefaultLossFunction` |  |
| `Engine` | Hardware-accelerated computation engine (CPU SIMD / GPU). |
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
| `Dispose(Boolean)` | Disposes the wrapped base model if it is disposable. |
| `GetActiveFeatureIndices` |  |
| `GetFeatureImportance` |  |
| `GetModelMetadata` |  |
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

