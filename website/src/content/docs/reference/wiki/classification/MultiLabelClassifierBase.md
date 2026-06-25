---
title: "MultiLabelClassifierBase<T>"
description: "Base class for multi-label classification models."
section: "API Reference"
---

`Base Classes` · `AiDotNet.Classification.MultiLabel`

Base class for multi-label classification models.

## For Beginners

This base class provides common functionality for multi-label
classifiers. Multi-label classification assigns multiple labels to each sample, unlike
traditional classification which assigns exactly one label.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MultiLabelClassifierBase(ClassifierOptions<>,IRegularization<,Matrix<>,Vector<>>)` | Initializes a new instance of the MultiLabelClassifierBase class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DefaultLossFunction` |  |
| `Engine` | Gets the hardware-accelerated computation engine for vectorized operations. |
| `FeatureNames` | Gets or sets the feature names. |
| `LabelNames` | Gets or sets the label names if available. |
| `NumClasses` | Gets or sets the number of classes (typically 2 for binary classification per label). |
| `NumFeatures` | Gets or sets the number of features. |
| `NumLabels` | Gets or sets the number of possible labels. |
| `NumOps` | Gets the numeric operations provider for type T. |
| `Options` | Gets the classifier options. |
| `ParameterCount` |  |
| `Regularization` | Gets the regularization method used to prevent overfitting. |
| `SupportsParameterInitialization` |  |
| `TaskType` | Gets or sets the classification task type. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyGradients(Vector<>,)` |  |
| `Clone` |  |
| `ComputeGradients(Matrix<>,Matrix<>,ILossFunction<>)` |  |
| `CreateNewInstance` | Gets the model type for this classifier. |
| `DeepCopy` |  |
| `Deserialize(Byte[])` |  |
| `DeserializeInternalUnchecked(Byte[])` | Internal, non-virtual, no-guard deserialization used by trusted framework call sites such as `DeepCopy`. |
| `Dispose` |  |
| `Dispose(Boolean)` | Releases resources held by this multi-label classifier. |
| `GetActiveFeatureIndices` |  |
| `GetDynamicShapeInfo` |  |
| `GetFeatureImportance` |  |
| `GetInputShape` |  |
| `GetModelMetadata` |  |
| `GetOptions` |  |
| `GetOutputShape` |  |
| `GetParameters` |  |
| `IsFeatureUsed(Int32)` |  |
| `LoadModel(String)` |  |
| `LoadState(Stream)` |  |
| `Predict(Matrix<>)` | Predicts binary label indicators for input samples. |
| `PredictMultiLabelProbabilities(Matrix<>)` | Core probability prediction implementation to be overridden by derived classes. |
| `PredictProbabilities(Matrix<>)` | Predicts label probabilities for input samples. |
| `SanitizeParameters(Vector<>)` |  |
| `SaveState(Stream)` |  |
| `Serialize` |  |
| `SerializeInternalUnchecked` | Internal, non-virtual, no-guard serialization used by trusted framework call sites such as `DeepCopy`. |
| `SetActiveFeatureIndices(IEnumerable<Int32>)` |  |
| `SetParameters(Vector<>)` |  |
| `ThrowIfDisposed` | Throws `ObjectDisposedException` if `Dispose` has already been called. |
| `Train(Matrix<>,Matrix<>)` | Trains the multi-label classifier. |
| `TrainMultiLabelCore(Matrix<>,Matrix<>)` | Core training implementation to be overridden by derived classes. |
| `WithParameters(Vector<>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `_defaultLossFunction` | The default loss function for this classifier. |

