---
title: "BalancedRandomForestClassifier<T>"
description: "Balanced Random Forest Classifier for imbalanced datasets."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Classification.ImbalancedEnsemble`

Balanced Random Forest Classifier for imbalanced datasets.

## For Beginners

When you have imbalanced data (e.g., 1000 normal transactions vs
10 fraudulent ones), regular classifiers often ignore the minority class. Balanced Random Forest
fixes this by training each tree on a balanced subset of data.

## How It Works

**How it works:** For each tree in the forest:

- Randomly sample from the minority class with replacement
- Randomly undersample the majority class to match minority count
- Train a decision tree on this balanced bootstrap

**Key advantages:**

- Better detection of minority class compared to standard Random Forest
- Maintains the ensemble benefits (reduced variance, robustness)
- No need to manually balance your dataset

**When to use:**

- Fraud detection where fraud cases are rare
- Medical diagnosis where positive cases are uncommon
- Any binary classification with significant class imbalance

**References:**

- Chen, C., Liaw, A., & Breiman, L. (2004). "Using Random Forest to Learn Imbalanced Data"

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `BalancedRandomForestClassifier(Int32,Nullable<Int32>,Nullable<Int32>,Int32,Int32,String,Boolean,Nullable<Int32>)` | Initializes a new instance of BalancedRandomForestClassifier. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyGradients(Vector<>,)` | Applies gradients (not applicable for tree models). |
| `BuildTree(Matrix<>,Vector<>,Int32[],Int32,Int32)` | Builds a decision tree recursively. |
| `ComputeGini(Vector<>,List<Int32>)` | Computes the Gini impurity for a set of samples. |
| `ComputeGradients(Matrix<>,Vector<>,ILossFunction<>)` | Computes gradients (not applicable for tree models). |
| `CountFeatureUsage(BalancedRandomForestClassifier<>.DecisionTreeNode,Double[])` | Counts feature usage in a tree. |
| `CreateBalancedBootstrap(Dictionary<Int32,List<Int32>>,Int32)` | Creates a balanced bootstrap sample. |
| `CreateNewInstance` | Creates a new instance of this model type. |
| `Deserialize(Byte[])` | Deserializes the trained model state including all decision trees. |
| `GetFeatureImportance` | Gets feature importance based on split usage. |
| `GetParameters` | Gets the model parameters. |
| `Predict(Matrix<>)` | Predicts class labels for the given input data. |
| `PredictTree(BalancedRandomForestClassifier<>.DecisionTreeNode,Matrix<>,Int32)` | Gets prediction from a single tree. |
| `Serialize` | Serializes the trained model state including all decision trees. |
| `SetParameters(Vector<>)` | Sets the model parameters. |
| `ShuffleArray([])` | Shuffles an array in place. |
| `Train(Matrix<>,Vector<>)` | Gets the model type. |
| `WithParameters(Vector<>)` | Creates a new instance with the specified parameters. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_bootstrap` | Whether to use bootstrap sampling (with replacement). |
| `_maxDepth` | Maximum depth of each tree. |
| `_maxFeatures` | Number of features to consider for best split. |
| `_minSamplesLeaf` | Minimum samples required in a leaf node. |
| `_minSamplesSplit` | Minimum samples required to split a node. |
| `_nEstimators` | Number of trees in the forest. |
| `_random` | Random number generator. |
| `_samplingStrategy` | Sampling strategy for the minority class. |
| `_trees` | The ensemble of decision trees. |

