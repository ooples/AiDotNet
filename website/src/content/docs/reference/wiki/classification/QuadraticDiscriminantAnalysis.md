---
title: "QuadraticDiscriminantAnalysis<T>"
description: "Quadratic Discriminant Analysis classifier."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Classification.DiscriminantAnalysis`

Quadratic Discriminant Analysis classifier.

## For Beginners

Quadratic Discriminant Analysis (QDA) is a classification technique that:

1. Models each class with its own Gaussian distribution
2. Each class has its own covariance matrix (unlike LDA)
3. Decision boundaries are quadratic (curved) instead of linear

When to use QDA over LDA:

- When classes have different covariance structures
- When you have enough samples per class to estimate covariance reliably
- When decision boundaries are naturally curved

Trade-offs:

- More flexible than LDA (can capture curved boundaries)
- Needs more parameters (separate covariance per class)
- More prone to overfitting with small datasets
- Computationally more expensive than LDA

## How It Works

QDA is similar to LDA but allows each class to have its own covariance matrix,
which creates quadratic (curved) decision boundaries.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `QuadraticDiscriminantAnalysis(DiscriminantAnalysisOptions<>,IRegularization<,Matrix<>,Vector<>>)` | Initializes a new instance of the QuadraticDiscriminantAnalysis class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Options` | Gets the QDA-specific options. |
| `SupportsParameterInitialization` | QDA computes parameters from per-class covariance matrices during training. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AddRegularization(Matrix<>)` | Adds regularization to the covariance matrix. |
| `ApplyGradients(Vector<>,)` |  |
| `Clone` |  |
| `ComputeClassCovariance(Matrix<>,Vector<>,Int32)` | Computes the covariance matrix for a specific class. |
| `ComputeClassMeans(Matrix<>,Vector<>)` | Computes the mean vector for each class. |
| `ComputeClassPriors(Vector<>)` | Computes class prior probabilities. |
| `ComputeGradients(Matrix<>,Vector<>,ILossFunction<>)` |  |
| `ComputeInverse(Matrix<>)` | Computes the inverse of a matrix using Gaussian elimination. |
| `ComputeLogDeterminant(Matrix<>)` | Computes the log determinant of a matrix using LU decomposition. |
| `CreateNewInstance` |  |
| `Deserialize(Byte[])` |  |
| `GetModelMetadata` |  |
| `GetParameters` |  |
| `Predict(Matrix<>)` |  |
| `PredictLogProbabilities(Matrix<>)` |  |
| `PredictProbabilities(Matrix<>)` |  |
| `Serialize` |  |
| `SetParameters(Vector<>)` |  |
| `Train(Matrix<>,Vector<>)` | Returns the model type identifier for this classifier. |
| `WithParameters(Vector<>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `_classCovarianceInverses` | Inverse of covariance matrix for each class. |
| `_classCovariances` | Covariance matrix for each class. |
| `_classLogDets` | Log determinant of covariance matrix for each class. |
| `_classMeans` | Class means for each class. |
| `_classPriors` | Class priors (prior probabilities). |

