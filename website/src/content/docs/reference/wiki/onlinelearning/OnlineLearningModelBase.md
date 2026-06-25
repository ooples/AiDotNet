---
title: "OnlineLearningModelBase<T>"
description: "Abstract base class for online (incremental) learning models."
section: "API Reference"
---

`Base Classes` · `AiDotNet.OnlineLearning`

Abstract base class for online (incremental) learning models.

## For Beginners

This class contains shared code that all online learning models need,
so each specific model doesn't have to reimplement it.

Key shared functionality:

- Tracking how many samples have been seen
- Managing the learning rate (step size for updates)
- Converting between single-sample and batch updates
- Standard IFullModel interface implementation

## How It Works

This base class provides common functionality for online learning models including
sample counting, learning rate scheduling, and incremental updates.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `OnlineLearningModelBase(Double,LearningRateSchedule)` | Initializes a new instance of the OnlineLearningModelBase class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DefaultLossFunction` | Gets the default loss function. |
| `FeatureNames` | Gets or sets the feature names. |
| `IsTrained` | Gets whether the model is trained (has seen at least one sample). |
| `NumFeatures` | Gets the number of features the model was trained on. |
| `ParameterCount` | Gets the total number of parameters in the model. |
| `SupportsParameterInitialization` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyGradients(Vector<>,)` | Applies gradients to update model parameters. |
| `Clone` | Creates a clone of the model. |
| `ComputeGradients(Matrix<>,Vector<>,ILossFunction<>)` | Computes gradients for the given input and target. |
| `CreateNewInstance` | Creates a new instance of the same type. |
| `DeepCopy` | Creates a deep copy of the model. |
| `Deserialize(Byte[])` | Deserializes the model from a byte array. |
| `DeserializeInternalUnchecked(Byte[])` | Internal, non-virtual, no-guard deserialization used by trusted framework call sites such as `DeepCopy`. |
| `Dispose` |  |
| `Dispose(Boolean)` | Releases resources held by this online-learning model. |
| `EnsureInitialized(Vector<>)` | Ensures the model is initialized, initializing if needed. |
| `GetActiveFeatureIndices` | Gets the indices of features that are actively used in the model. |
| `GetDynamicShapeInfo` |  |
| `GetFeatureImportance` | Gets the feature importance scores. |
| `GetInputShape` | Saves the model to a file. |
| `GetLearningRate` | Gets the current learning rate based on the schedule. |
| `GetModelMetadata` | Gets the model type. |
| `GetOutputShape` |  |
| `GetParameters` | Gets all model parameters as a single vector. |
| `GetSampleCount` | Gets the number of samples the model has seen. |
| `Initialize(Int32)` | Initializes the model for a given number of features. |
| `IsFeatureUsed(Int32)` | Determines whether a specific feature is used in the model. |
| `LoadModel(String)` | Loads the model from a file. |
| `LoadState(Stream)` | Loads the model's state from a stream. |
| `PartialFit(Matrix<>,Vector<>)` | Updates the model with a mini-batch of training examples. |
| `PartialFit(Vector<>,)` | Updates the model with a single training example. |
| `Predict(Matrix<>)` | Standard prediction - returns predictions for all samples. |
| `PredictSingle(Vector<>)` | Predicts the target value for a single sample. |
| `Reset` | Resets the model to its initial state. |
| `SanitizeParameters(Vector<>)` |  |
| `SaveState(Stream)` | Saves the model's state to a stream. |
| `Serialize` | Serializes the model to a byte array. |
| `SerializeInternalUnchecked` | Internal, non-virtual, no-guard serialization used by trusted framework call sites such as `DeepCopy`. |
| `SetActiveFeatureIndices(IEnumerable<Int32>)` | Sets the active feature indices for this model. |
| `SetParameters(Vector<>)` | Sets the parameters for this model. |
| `Train(Matrix<>,Vector<>)` | Standard model training - equivalent to PartialFit. |
| `WithParameters(Vector<>)` | Creates a new instance of the model with specified parameters. |

## Fields

| Field | Summary |
|:-----|:--------|
| `InitialLearningRate` | Initial learning rate. |
| `IsInitialized` | Indicates whether the model has been initialized. |
| `LearningRateScheduleType` | Learning rate decay schedule type. |
| `NumOps` | Numeric operations helper for generic math. |
| `SampleCount` | Number of samples the model has been trained on. |
| `_defaultLossFunction` | The default loss function for gradient computation. |

