---
title: "DecisionTreeNode<T>"
description: "Represents a node in a decision tree for machine learning algorithms."
section: "API Reference"
---

`Helpers & Utilities` · `AiDotNet.LinearAlgebra`

Represents a node in a decision tree for machine learning algorithms.

## For Beginners

Think of a decision tree like a flowchart of questions. Starting at the top (root),
each question (node) splits the data based on a feature (like "Is temperature > 70°F?").
Following the answers (branches) leads you to more questions or eventually to a final answer (leaf node).
Decision trees are popular because they're easy to understand and visualize - they make decisions
similar to how humans think.

## How It Works

A decision tree is a flowchart-like structure where each internal node represents a decision based on a feature,
each branch represents an outcome of that decision, and each leaf node represents a prediction or classification.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DecisionTreeNode` | Initializes a new instance of the `DecisionTreeNode` class as a leaf node. |
| `DecisionTreeNode()` | Initializes a new instance of the `DecisionTreeNode` class as a leaf node with a prediction. |
| `DecisionTreeNode(Int32,)` | Initializes a new instance of the `DecisionTreeNode` class as an internal decision node. |

## Properties

| Property | Summary |
|:-----|:--------|
| `FeatureIndex` | Gets or sets the index of the feature used for splitting at this node. |
| `IsLeaf` | Gets or sets a value indicating whether this node is a leaf node (has no children). |
| `Left` | Gets or sets the left child node (typically represents the "less than" or "no" branch). |
| `LeftSampleCount` | Gets or sets the number of samples that went to the left child after splitting. |
| `LinearModel` | Gets or sets the linear regression model for this node (used in some advanced tree variants). |
| `NodeImportance` | Precomputed feature-importance contribution of this split (variance reduction × node sample count), set by the index-based regression builder so importances don't require every node to retain its full sample set. |
| `NodeImpurity` | Precomputed impurity (population variance of the targets) at this node, set by the index-based regression builder. |
| `Prediction` | Gets or sets the prediction value for this node when it's a leaf node. |
| `Predictions` | Gets or sets the vector of predictions for samples at this node. |
| `Right` | Gets or sets the right child node (typically represents the "greater than or equal to" or "yes" branch). |
| `RightSampleCount` | Gets or sets the number of samples that went to the right child after splitting. |
| `SampleValues` | Gets or sets the list of target values from the samples at this node. |
| `Samples` | Gets or sets the list of data samples that reached this node during training. |
| `SplitValue` | Gets or sets the value used to split the data at this node. |
| `SumSquaredError` | Gets or sets the sum of squared errors for predictions at this node. |
| `Threshold` | Gets or sets the threshold value used to determine the split direction. |
| `_numOps` | Gets or sets the numeric operations helper for the generic type T. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateSumSquaredError` | Calculates the sum of squared errors for the predictions at this node. |
| `UpdateNodeStatistics` | Updates statistical information for this node based on its samples. |

