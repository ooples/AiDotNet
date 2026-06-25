---
title: "NonLinearRegressionBase<T>"
description: "Base class for non-linear regression algorithms that provides common functionality for training and prediction."
section: "API Reference"
---

`Base Classes` · `AiDotNet.Regression`

Base class for non-linear regression algorithms that provides common functionality for training and prediction.

## How It Works

This abstract class implements core functionality shared by different non-linear regression algorithms,
including kernel functions, regularization, and model serialization/deserialization.

Non-linear regression models can capture complex relationships in data that linear models cannot represent.
They typically use kernel functions to transform the input space into a higher-dimensional feature space
where the relationship becomes linear.

For Beginners:
Non-linear regression is used when your data doesn't follow a straight line pattern. These models can
capture curved or complex relationships between your input features and target values. Think of it like
having a flexible curve that can bend and shape itself to fit your data points, rather than just a
straight line.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `NonLinearRegressionBase(NonLinearRegressionOptions,IRegularization<,Matrix<>,Vector<>>,ILossFunction<>)` | Initializes a new instance of the NonLinearRegressionBase class with the specified options and regularization. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Alphas` | Gets or sets the alpha coefficients for each support vector. |
| `B` | Gets or sets the bias term (intercept) of the model. |
| `DefaultLossFunction` |  |
| `Engine` | Gets the global execution engine for vector operations. |
| `FeatureNames` | Gets or sets the feature names. |
| `NumOps` | Gets the numeric operations provider for the specified type T. |
| `Options` | Gets the configuration options for the non-linear regression model. |
| `Regularization` | Gets the regularization method used to prevent overfitting. |
| `SupportVectors` | Gets or sets the support vectors used by the model. |
| `SupportsParameterInitialization` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyGradients(Vector<>,)` |  |
| `Clip(,,)` | Clips a value to be within the specified range. |
| `Clone` | Creates a shallow copy of the model. |
| `ComputeGradients(Matrix<>,Vector<>,ILossFunction<>)` |  |
| `CreateInstance` | Creates a new instance of the derived model class. |
| `DeepCopy` | Creates a deep copy of the model. |
| `Deserialize(Byte[])` | Deserializes the model from a byte array. |
| `Dispose` |  |
| `Dispose(Boolean)` | Releases resources held by this nonlinear regressor. |
| `ExtractModelParameters` | Extracts the support vectors and their coefficients after optimization. |
| `GetActiveFeatureIndices` | Gets the indices of features that are actively used by the model. |
| `GetDynamicShapeInfo` |  |
| `GetFeatureImportance` | Gets the feature importance scores as a dictionary. |
| `GetInputShape` |  |
| `GetModelMetadata` | Gets metadata about the model. |
| `GetOptions` |  |
| `GetOutputShape` |  |
| `GetParameters` | Gets the model parameters as a single vector. |
| `InitializeModel(Matrix<>,Vector<>)` | Initializes the model parameters before optimization. |
| `IsFeatureUsed(Int32)` | Determines whether a specific feature is used by the model. |
| `KernelFunction(Vector<>,Vector<>)` | Computes the kernel function between two vectors. |
| `LoadState(Stream)` | Loads the model's state from a stream. |
| `OptimizeModel(Matrix<>,Vector<>)` | Optimizes the model parameters using the training data. |
| `Predict(Matrix<>)` | Makes predictions for the given input data. |
| `PredictSingle(Vector<>)` | Makes a prediction for a single input example. |
| `SanitizeParameters(Vector<>)` |  |
| `SaveState(Stream)` | Saves the model's current state to a stream. |
| `Serialize` | Gets the type of the model. |
| `SetActiveFeatureIndices(IEnumerable<Int32>)` | Sets the active feature indices for this model. |
| `SetParameters(Vector<>)` | Sets the parameters for this model. |
| `Train(Matrix<>,Vector<>)` | Trains the non-linear regression model on the provided data. |
| `ValidateInputs(Matrix<>,Vector<>)` | Validates the input data before training. |
| `WithParameters(Vector<>)` | Creates a new model with the specified parameters. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_defaultLossFunction` | Gets the default loss function for this non-linear regression model. |

