---
title: "SupportVectorClassifier<T>"
description: "Support Vector Classifier using kernel methods for non-linear classification."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Classification.SVM`

Support Vector Classifier using kernel methods for non-linear classification.

## For Beginners

The Support Vector Classifier (SVC) finds the best boundary between classes by:

1. Looking at all training points
2. Finding the points closest to the decision boundary (support vectors)
3. Drawing a boundary that maximizes the margin to these support vectors
4. Using a kernel trick to handle non-linear boundaries

Common use cases:

- Text classification (spam detection, sentiment analysis)
- Image classification
- Bioinformatics (protein classification)
- Any problem with clear separation between classes

## How It Works

This implementation uses a simplified Sequential Minimal Optimization (SMO) algorithm
to find the optimal separating hyperplane in the kernel-induced feature space.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SupportVectorClassifier(SVMOptions<>,IRegularization<,Matrix<>,Vector<>>)` | Initializes a new instance of the SupportVectorClassifier class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ClipAlpha(,,)` | Clips alpha to bounds [L, H]. |
| `Clone` |  |
| `ComputeDecision(Vector<>)` | Computes the decision value for a sample (Vector overload — used by Predict on user-supplied inputs). |
| `ComputeDecisionFromArray([])` | Array-typed decision evaluation. |
| `ComputeError(Int32)` | Computes the prediction error for sample i. |
| `ComputeKernelCached(Int32,Int32)` | Computes kernel value with caching support. |
| `ComputeKernelFromArrays([],[])` | Dispatches the configured kernel to its array overload. |
| `CreateNewInstance` |  |
| `DecisionFunction(Matrix<>)` |  |
| `Deserialize(Byte[])` |  |
| `ExtractSupportVectors` | Extracts support vectors from training data. |
| `Max(,)` | Returns the maximum of two values. |
| `Min(,)` | Returns the minimum of two values. |
| `PredictProbabilities(Matrix<>)` |  |
| `SelectSecondAlpha(Int32,Int32)` | Selects the second alpha for SMO. |
| `Serialize` |  |
| `Sigmoid()` | Sigmoid function for probability estimation. |
| `Train(Matrix<>,Vector<>)` | Returns the model type identifier for this classifier. |
| `TrainSMO` | Simplified SMO algorithm for training. |
| `UpdateIntercept(Int32,Int32,,,,,,,)` | Updates the intercept after alpha update. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_alphas` | Alpha coefficients from SMO algorithm. |
| `_alphasArr` | Cached alpha coefficients as a raw array, refreshed alongside `_alphas` after each SMO inner-loop update. |
| `_random` | Random number generator for SMO. |
| `_xTrain` | Stored training features. |
| `_xTrainRows` | Cached training rows as raw arrays. |
| `_yTrain` | Stored training labels (converted to +1/-1 for binary). |
| `_yTrainArr` | Cached training labels as a raw array (same rationale as `_xTrainRows`). |

