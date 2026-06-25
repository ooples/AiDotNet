---
title: "LinearDiscriminantAnalysis<T>"
description: "Linear Discriminant Analysis classifier."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Classification.DiscriminantAnalysis`

Linear Discriminant Analysis classifier.

## For Beginners

Linear Discriminant Analysis (LDA) is a classification technique that:

1. Finds the directions that best separate the classes
2. Projects the data onto these directions
3. Classifies based on which class region the projection falls into

Key assumptions:

- Each class follows a Gaussian distribution
- All classes share the same covariance matrix
- Features are continuous

When to use LDA:

- When classes are well-separated
- When you have more samples than features
- When you want a simple, interpretable classifier
- As a dimensionality reduction technique

Trade-offs:

- Assumes shared covariance (use QDA if classes differ)
- Linear boundaries (may not work for complex patterns)
- Sensitive to outliers

## How It Works

LDA finds a linear combination of features that best separates classes.
It projects data onto a lower-dimensional space while maximizing class separation.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LinearDiscriminantAnalysis(DiscriminantAnalysisOptions<>,IRegularization<,Matrix<>,Vector<>>)` | Initializes a new instance of the LinearDiscriminantAnalysis class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Options` | Gets the LDA-specific options. |
| `SupportsParameterInitialization` | LDA computes parameters from class covariance matrices during training. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AddRegularization(Matrix<>)` | Adds regularization to the covariance matrix. |
| `ApplyGradients(Vector<>,)` |  |
| `Clone` |  |
| `ComputeClassMeans(Matrix<>,Vector<>)` | Computes the mean vector for each class. |
| `ComputeClassPriors(Vector<>)` | Computes class prior probabilities. |
| `ComputeGradients(Matrix<>,Vector<>,ILossFunction<>)` |  |
| `ComputeInverse(Matrix<>)` | Computes the inverse of a matrix using Gaussian elimination. |
| `ComputePooledCovariance(Matrix<>,Vector<>)` | Computes the pooled within-class covariance matrix. |
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
| `_classMeans` | Class means for each class. |
| `_classPriors` | Class priors (prior probabilities). |
| `_covarianceInverse` | Inverse of the pooled covariance matrix. |
| `_pooledCovariance` | Pooled within-class covariance matrix (shared by all classes). |

