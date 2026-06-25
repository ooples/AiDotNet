---
title: "ClassifierBase<T>"
description: "Provides a base implementation for classification algorithms that predict categorical outcomes."
section: "API Reference"
---

`Base Classes` · `AiDotNet.Classification`

Provides a base implementation for classification algorithms that predict categorical outcomes.

## For Beginners

Classification is about predicting which category something belongs to.
This base class provides the foundation for different classification techniques, handling
common operations like making predictions and saving/loading models. Think of it as
a template that specific classification algorithms can customize while reusing the shared
functionality.

## How It Works

This abstract class implements common functionality for classification models, including
prediction, serialization/deserialization, and parameter management. Specific classification
algorithms should inherit from this class and implement the Train and Predict methods.

The class supports various options like class weighting to handle imbalanced datasets
and different classification task types (binary, multi-class, multi-label, ordinal).

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ClassifierBase(ClassifierOptions<>,IRegularization<,Matrix<>,Vector<>>,ILossFunction<>)` | Initializes a new instance of the ClassifierBase class with the specified options and regularization. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ClassLabels` | Gets or sets the class labels learned during training. |
| `DefaultLossFunction` | Gets the total number of parameters in the model. |
| `Engine` | Gets the global execution engine for vector operations. |
| `ExpectedParameterCount` | Gets the expected number of parameters for this model. |
| `FeatureNames` | Gets or sets the feature names. |
| `NumClasses` | Gets or sets the number of classes in the classification problem. |
| `NumFeatures` | Gets or sets the number of features expected by this classifier. |
| `NumOps` | Gets the numeric operations for the specified type T. |
| `Options` | Gets the classifier options. |
| `ParameterCount` | Gets the expected number of parameters for this classifier. |
| `Regularization` | Gets the regularization method used to prevent overfitting. |
| `SupportsParameterInitialization` | Whether this classifier supports parameter initialization. |
| `TaskType` | Gets or sets the type of classification task. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clone` | Creates a clone of the classifier model. |
| `ComputeClassWeights(Vector<>)` | Computes class weights for handling imbalanced datasets. |
| `CreateNewInstance` | Creates a new instance of the same type as this classifier. |
| `DeepCopy` | Creates a deep copy of the classifier model. |
| `Deserialize(Byte[])` | Deserializes the model from a byte array. |
| `DeserializeInternalUnchecked(Byte[])` | Internal, non-virtual, no-guard deserialization used by trusted framework call sites such as `DeepCopy`. |
| `Dispose` |  |
| `Dispose(Boolean)` | Releases resources held by this classifier. |
| `ExtractClassLabels(Vector<>)` | Extracts unique class labels from the training data. |
| `GetActiveFeatureIndices` | Gets all model parameters as a single vector. |
| `GetClassIndexFromLabel()` | Gets the class index for a given label value. |
| `GetDynamicShapeInfo` |  |
| `GetFeatureImportance` | Gets the feature importance scores as a dictionary. |
| `GetInputShape` |  |
| `GetModelMetadata` | Gets metadata about the model. |
| `GetOptions` |  |
| `GetOutputShape` |  |
| `InferTaskType(Vector<>)` | Infers the classification task type from the training labels. |
| `IsFeatureUsed(Int32)` | Determines whether a specific feature is used in the model. |
| `LoadModel(String)` | Loads a classifier model from a file. |
| `LoadState(Stream)` | Loads the model's state from a stream. |
| `Predict(Matrix<>)` | Predicts class labels for the given input data. |
| `SanitizeParameters(Vector<>)` | Sanitizes parameters to ensure they satisfy model constraints. |
| `SaveState(Stream)` | Saves the model's current state to a stream. |
| `Serialize` | Gets the type of the model. |
| `SerializeInternalUnchecked` | Internal, non-virtual, no-guard serialization used by trusted framework call sites such as `DeepCopy`. |
| `SetActiveFeatureIndices(IEnumerable<Int32>)` | Sets the active feature indices for this model. |
| `Train(Matrix<>,Vector<>)` | Trains the classifier on the provided data. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_defaultLossFunction` | Gets the default loss function for this classifier. |

