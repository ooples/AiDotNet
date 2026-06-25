---
title: "BalancedBaggingClassifier<T>"
description: "Balanced Bagging Classifier for imbalanced datasets."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Classification.ImbalancedEnsemble`

Balanced Bagging Classifier for imbalanced datasets.

## For Beginners

Bagging (Bootstrap Aggregating) creates diverse classifiers by
training each on a random bootstrap sample. Balanced Bagging adds undersampling to ensure
each bootstrap is class-balanced, improving minority class detection.

## How It Works

**How it works:** For each base classifier:

- Sample minority class with replacement (bootstrap)
- Undersample majority class to match minority count
- Train a base classifier (typically decision tree) on balanced bootstrap
- Combine predictions using majority voting

**Key advantages:**

- **Reduces variance:** Multiple diverse classifiers give stable predictions
- **Handles imbalance:** Each classifier sees balanced data
- **Parallelizable:** Each base classifier can be trained independently
- **Flexible:** Works with any base classifier, not just trees

**Difference from BalancedRandomForest:** BalancedBagging can use any base classifier
and each classifier sees all features, while BalancedRandomForest uses trees with random
feature subsets at each split.

**When to use:**

- When you want ensemble benefits with imbalanced data
- When you want to use a specific base classifier (not just trees)
- When variance reduction is more important than bias reduction

**References:**

- Hido, S., & Kashima, H. (2009). "Roughly Balanced Bagging for Imbalanced Data"
- Wang, S., & Yao, X. (2009). "Diversity Analysis on Imbalanced Data Sets"

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `BalancedBaggingClassifier(Int32,Nullable<Int32>,Int32,Int32,Double,Boolean,Nullable<Int32>)` | Initializes a new instance of BalancedBaggingClassifier. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyGradients(Vector<>,)` | Applies gradients (not applicable for tree ensembles). |
| `BuildTree(Matrix<>,Vector<>,Int32[],Int32)` | Builds a decision tree recursively. |
| `ComputeGini(Vector<>,List<Int32>)` | Computes Gini impurity for a set of samples. |
| `ComputeGradients(Matrix<>,Vector<>,ILossFunction<>)` | Computes gradients (not applicable for tree ensembles). |
| `CountFeatureUsage(BalancedBaggingClassifier<>.DecisionTreeNode,Double[])` | Counts feature usage in a tree. |
| `CreateBalancedBootstrap(Dictionary<Int32,List<Int32>>,Int32,Int32)` | Creates a balanced bootstrap sample. |
| `CreateNewInstance` | Creates a new instance of this model type. |
| `Deserialize(Byte[])` | Deserializes the trained model state including all base classifier trees. |
| `GetFeatureImportance` | Gets feature importance based on split usage. |
| `GetParameters` | Gets the model parameters. |
| `Predict(Matrix<>)` | Predicts class labels for the given input data. |
| `PredictTree(BalancedBaggingClassifier<>.DecisionTreeNode,Matrix<>,Int32)` | Gets prediction from a single tree. |
| `Serialize` | Serializes the trained model state including all base classifier trees. |
| `SetParameters(Vector<>)` | Sets the model parameters. |
| `Train(Matrix<>,Vector<>)` | Gets the model type. |
| `WithParameters(Vector<>)` | Creates a new instance with the specified parameters. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_baseClassifiers` | The ensemble of base classifiers. |
| `_bootstrapMinority` | Whether to use bootstrap sampling for minority class. |
| `_maxDepth` | Maximum depth of decision tree base classifiers. |
| `_minSamplesLeaf` | Minimum samples required in a leaf node. |
| `_minSamplesSplit` | Minimum samples required to split a node. |
| `_nEstimators` | Number of base classifiers in the ensemble. |
| `_random` | Random number generator. |
| `_samplingRatio` | Ratio of majority samples to minority samples. |

