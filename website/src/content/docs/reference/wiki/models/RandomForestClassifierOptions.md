---
title: "RandomForestClassifierOptions<T>"
description: "Configuration options for Random Forest classifiers."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for Random Forest classifiers.

## For Beginners

Random Forest is like asking many decision trees for their opinion!

Imagine you're trying to classify a flower species:

1. Create many decision trees, each trained on a random subset of your data
2. Each tree considers only a random subset of features at each split
3. To classify a new flower, ask all trees and take a vote

Why does this work so well?

- Each tree is slightly different due to random sampling
- Averaging many "weak" trees often creates a "strong" classifier
- It's much harder to overfit than a single deep tree

Random Forest is one of the most popular algorithms because it:

- Works well with default settings
- Handles both classification and regression
- Is robust to outliers and noise
- Provides feature importance scores

## How It Works

Random Forest is an ensemble learning method that constructs multiple decision trees
during training and outputs the class that is the mode (most frequent) of the classes
predicted by individual trees.

## Properties

| Property | Summary |
|:-----|:--------|
| `Bootstrap` | Gets or sets whether to use bootstrap sampling. |
| `Criterion` | Gets or sets the criterion used to measure the quality of a split. |
| `MaxDepth` | Gets or sets the maximum depth of each tree. |
| `MaxFeatures` | Gets or sets the number of features to consider when looking for the best split. |
| `MinImpurityDecrease` | Gets or sets the minimum impurity decrease required for a split. |
| `MinSamplesLeaf` | Gets or sets the minimum number of samples required at a leaf node. |
| `MinSamplesSplit` | Gets or sets the minimum number of samples required to split an internal node. |
| `NEstimators` | Gets or sets the number of trees in the forest. |
| `NJobs` | Gets or sets the number of jobs for parallel training. |
| `OobScore` | Gets or sets whether to compute out-of-bag score during training. |

