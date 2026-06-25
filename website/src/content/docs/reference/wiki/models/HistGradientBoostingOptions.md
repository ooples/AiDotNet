---
title: "HistGradientBoostingOptions"
description: "Configuration options for Histogram-based Gradient Boosting, a fast ensemble learning technique that uses binned features for efficient tree building."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for Histogram-based Gradient Boosting, a fast ensemble learning technique
that uses binned features for efficient tree building.

## For Beginners

Think of Histogram Gradient Boosting as a smarter, faster version
of regular gradient boosting. Instead of checking every possible split point for each feature
(which can be millions of operations), it first groups similar values into "bins" (like putting
temperatures into ranges: 0-10, 10-20, 20-30, etc.). Then it only needs to check splits between
bins, which is much faster.

This approach:

- Trains 10-100x faster than traditional gradient boosting on large datasets
- Uses less memory by storing bin indices instead of full feature values
- Often achieves similar or better accuracy due to the regularization effect of binning
- Handles missing values natively without imputation

Use Histogram Gradient Boosting when:

- You have a large dataset (millions of rows)
- Training time is a concern
- You want state-of-the-art performance on tabular data

## How It Works

Histogram-based Gradient Boosting is an optimization of traditional gradient boosting that
discretizes continuous features into a small number of bins. This dramatically speeds up the
algorithm by reducing the number of split points to consider, similar to algorithms like
LightGBM and XGBoost's "hist" tree method.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `HistGradientBoostingOptions` | Parameterless constructor used when options are built up via object initializers. |
| `HistGradientBoostingOptions(HistGradientBoostingOptions)` | Copy constructor. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ColsampleByTree` | Gets or sets the fraction of features to consider for each split. |
| `EarlyStoppingRounds` | Gets or sets the number of rounds to wait for improvement before stopping. |
| `L2Regularization` | Gets or sets the L2 regularization parameter. |
| `LearningRate` | Gets or sets the learning rate (shrinkage factor). |
| `MaxBins` | Gets or sets the maximum number of bins for discretizing feature values. |
| `MaxDepth` | Gets or sets the maximum depth of each tree. |
| `MaxLeafNodes` | Gets or sets the maximum number of leaf nodes per tree. |
| `MinGainToSplit` | Gets or sets the minimum improvement required to make a split. |
| `MinSamplesLeaf` | Gets or sets the minimum number of samples required at a leaf node. |
| `NumberOfIterations` | Gets or sets the number of boosting iterations (trees). |
| `SubsampleRatio` | Gets or sets the fraction of samples to use for each tree (row subsampling). |
| `UseEarlyStopping` | Gets or sets whether to use early stopping to prevent overfitting. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clone` | Returns a deep copy of this options instance. |

