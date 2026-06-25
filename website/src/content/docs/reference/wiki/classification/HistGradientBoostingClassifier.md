---
title: "HistGradientBoostingClassifier<T>"
description: "Histogram-based Gradient Boosting Classifier."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Classification.Boosting`

Histogram-based Gradient Boosting Classifier.

## For Beginners

This is a fast gradient boosting implementation that uses histograms
to speed up the tree-building process. It's similar to LightGBM and scikit-learn's
HistGradientBoostingClassifier.

## How It Works

**How it works:** Instead of evaluating all possible split points, this algorithm:

- Bins continuous features into a fixed number of buckets (histograms)
- Builds trees by evaluating splits only at bin boundaries
- Uses gradient boosting to iteratively improve predictions

**Key advantages:**

- **Speed:** Much faster than traditional gradient boosting on large datasets
- **Memory:** Uses less memory due to binning
- **Missing values:** Please impute before training (missing-value handling is not built in)
- **Scalability:** Scales well to millions of samples

**When to use:**

- When you have a large dataset (thousands to millions of samples)
- When you need fast training without sacrificing much accuracy
- For tabular data classification problems

**References:**

- Ke, G. et al. (2017). "LightGBM: A Highly Efficient Gradient Boosting Decision Tree"
- Scikit-learn HistGradientBoostingClassifier implementation

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `HistGradientBoostingClassifier(Int32,Int32,Int32,Double,Int32,Double,Nullable<Int32>)` | Initializes a new instance of HistGradientBoostingClassifier. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyGradients(Vector<>,)` | Applies gradients to update model parameters. |
| `BinFeatures(Matrix<>,Double[][])` | Bins features using precomputed boundaries. |
| `BuildHistTree(Int32[0:,0:],Double[],Double[],Int32,Int32[])` | Builds a histogram-based decision tree. |
| `Clone` |  |
| `ComputeBinBoundaries(Matrix<>)` | Computes histogram bin boundaries for each feature. |
| `ComputeGradients(Matrix<>,Vector<>,ILossFunction<>)` | Computes gradients for the model parameters. |
| `ComputeGradientsAndHessians(Vector<>,Double[0:,0:],Int32)` | Computes gradients and hessians for gradient boosting. |
| `ComputeInitialPrediction(Vector<>)` | Computes the initial prediction (prior). |
| `CountFeatureUsage(HistGradientBoostingClassifier<>.HistTree,Double[])` | Counts feature usage in a tree for importance calculation. |
| `CreateNewInstance` | Creates a new instance of this model type. |
| `Deserialize(Byte[])` |  |
| `GetFeatureImportance` | Gets feature importance based on total gain reduction. |
| `GetParameters` | Gets the model parameters. |
| `Predict(Matrix<>)` | Predicts class labels for the given input data. |
| `PredictTree(HistGradientBoostingClassifier<>.HistTree,Int32[0:,0:],Int32)` | Gets the prediction from a tree for a single sample. |
| `Serialize` |  |
| `SetParameters(Vector<>)` | Sets the model parameters. |
| `Train(Matrix<>,Vector<>)` | Gets the model type. |
| `WithParameters(Vector<>)` | Creates a new instance with the specified parameters. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_binBoundaries` | The bin boundaries for each feature. |
| `_initialPrediction` | The initial prediction (log-odds for binary, class probabilities for multiclass). |
| `_l2Regularization` | L2 regularization strength. |
| `_learningRate` | Learning rate shrinkage. |
| `_maxBins` | Number of histogram bins per feature. |
| `_maxDepth` | Maximum depth of each tree. |
| `_minSamplesLeaf` | Minimum samples required to split a node. |
| `_nEstimators` | Number of boosting iterations (trees). |
| `_random` | Random number generator. |
| `_trees` | The ensemble of histogram-based decision trees. |

