---
title: "DARTOptions"
description: "Configuration options for DART (Dropouts meet Multiple Additive Regression Trees)."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for DART (Dropouts meet Multiple Additive Regression Trees).

## For Beginners

Regular gradient boosting can overfit by making each new tree
perfectly complement all previous trees. DART introduces randomness by "dropping out"
some trees during training:

- Imagine you have a team of experts (trees) making predictions together
- In regular boosting, each new expert learns to fix exactly what the whole team got wrong
- In DART, when training a new expert, some existing experts are temporarily removed
- This forces the new expert to be more versatile, not just filling a specific gap
- The result is a more robust ensemble that generalizes better to new data

DART is especially useful when:

- You're overfitting with regular gradient boosting
- You want more robust predictions
- You have enough computational budget (DART is slower than regular boosting)

## How It Works

DART applies the dropout concept from neural networks to gradient boosting. During each
boosting iteration, a random subset of existing trees is dropped, and the new tree is fit
to the residuals considering only the non-dropped trees. This helps prevent overfitting
and improves generalization.

Reference: Rashmi, K.V. & Gilad-Bachrach, R. (2015). "DART: Dropouts meet Multiple
Additive Regression Trees". AISTATS 2015.

## Properties

| Property | Summary |
|:-----|:--------|
| `DropoutMode` | Gets or sets the dropout mode. |
| `DropoutRate` | Gets or sets the dropout rate (probability of dropping a tree). |
| `FeatureFraction` | Gets or sets the fraction of features to consider for each split. |
| `L2Regularization` | Gets or sets the L2 regularization strength for leaf weights. |
| `LearningRate` | Gets or sets the learning rate (shrinkage factor). |
| `MaxDepth` | Gets or sets the maximum depth of each tree. |
| `MinSamplesLeaf` | Gets or sets the minimum number of samples required to split a node. |
| `MinSplitGain` | Gets or sets the minimum loss reduction required to make a split. |
| `Normalization` | Gets or sets the normalization strategy after dropout. |
| `NumberOfIterations` | Gets or sets the number of boosting iterations (trees to build). |
| `OneDrop` | Gets or sets whether to use one-drop strategy. |
| `SampleDroppedTreesProbability` | Gets or sets the probability of adding a dropped tree back to the ensemble. |
| `Seed` | Gets or sets the random seed for reproducibility. |
| `SkipDropoutIterations` | Gets or sets whether to skip dropout during the first k iterations. |
| `SubsampleFraction` | Gets or sets the fraction of samples to use for each tree. |

