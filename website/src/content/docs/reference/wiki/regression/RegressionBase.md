---
title: "RegressionBase<T>"
description: "Provides a base implementation for regression algorithms that model the relationship between a dependent variable and one or more independent variables."
section: "API Reference"
---

`Base Classes` · `AiDotNet.Regression`

Provides a base implementation for regression algorithms that model the relationship
between a dependent variable and one or more independent variables.

## For Beginners

Regression is a statistical method for modeling the relationship between variables.
This base class provides the foundation for different regression techniques, handling
common operations like making predictions and saving/loading models. Think of it as
a template that specific regression algorithms can customize while reusing the shared
functionality.

## How It Works

This abstract class implements common functionality for regression models, including
prediction, serialization/deserialization, and solving linear systems. Specific regression
algorithms should inherit from this class and implement the Train method.

The class supports various options like regularization to prevent overfitting and
different decomposition methods for solving linear systems.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `RegressionBase(RegressionOptions<>,IRegularization<,Matrix<>,Vector<>>,ILossFunction<>)` | Initializes a new instance of the RegressionBase class with the specified options and regularization. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Coefficients` | Gets or sets the coefficients (weights) of the regression model. |
| `DefaultLossFunction` |  |
| `Engine` | Gets the global execution engine for vector operations. |
| `ExpectedParameterCount` | Gets the expected number of parameters (coefficients plus intercept if used). |
| `FeatureNames` | Gets or sets the feature names. |
| `HasIntercept` | Gets a value indicating whether the model includes an intercept term. |
| `Intercept` | Gets or sets the intercept (bias) term of the regression model. |
| `NumOps` | Gets the numeric operations for the specified type T. |
| `Options` | Gets the regression options. |
| `Regularization` | Gets the regularization method used to prevent overfitting. |
| `SupportsParameterInitialization` |  |
| `TrainingFeatureCount` | Gets or sets the number of input features seen during training. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyGradients(Vector<>,)` |  |
| `CalculateFeatureImportances` | Gets the type of the model. |
| `Clone` | Creates a clone of the regression model. |
| `ComputeGradients(Matrix<>,Vector<>,ILossFunction<>)` |  |
| `CreateNewInstance` | Creates a new instance of the same type as this neural network. |
| `DeepCopy` | Creates a deep copy of the regression model. |
| `Deserialize(Byte[])` | Deserializes the model from a byte array. |
| `DeserializeInternalUnchecked(Byte[])` | Internal, non-virtual, no-guard deserialization used by trusted framework call sites such as `DeepCopy`. |
| `Dispose` |  |
| `Dispose(Boolean)` | Releases resources held by this regressor. |
| `GetActiveFeatureIndices` | Gets the indices of features that are actively used in the model. |
| `GetDynamicShapeInfo` |  |
| `GetFeatureImportance` | Gets the feature importance scores as a dictionary. |
| `GetInputShape` | Saves the regression model to a file. |
| `GetModelMetadata` | Gets metadata about the model. |
| `GetOptions` |  |
| `GetOutputShape` |  |
| `GetParameters` | Gets all model parameters (coefficients and intercept) as a single vector. |
| `IsFeatureUsed(Int32)` | Determines whether a specific feature is used in the model. |
| `LoadModel(String)` | Loads a regression model from a file. |
| `LoadState(Stream)` | Loads the model's state from a stream. |
| `Predict(Matrix<>)` | Makes predictions for the given input data. |
| `SanitizeParameters(Vector<>)` |  |
| `SaveState(Stream)` | Saves the model's current state to a stream. |
| `Serialize` | Serializes the model to a byte array. |
| `SerializeInternalUnchecked` | Internal, non-virtual, no-guard serialization used by trusted framework call sites such as `DeepCopy`. |
| `SetActiveFeatureIndices(IEnumerable<Int32>)` | Sets the active feature indices for this model. |
| `SetCoefficientsAndIntercept(Vector<>,)` | Sets the coefficients and intercept directly for deserialization purposes. |
| `SetParameters(Vector<>)` | Sets the parameters for this model. |
| `SolveNormalEquation(Matrix<>,Vector<>)` | Solves a linear system using the normal equation. |
| `SolveSystem(Matrix<>,Vector<>)` | Solves a linear system of equations using the specified decomposition method. |
| `Train(Matrix<>,Vector<>)` | Trains the regression model on the provided data. |
| `WithParameters(Vector<>)` | Creates a new instance of the model with specified parameters. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_defaultLossFunction` | Gets the default loss function for this regression model. |

