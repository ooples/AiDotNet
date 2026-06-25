---
title: "DARTClassifierOptions<T>"
description: "Configuration options for DART (Dropouts meet Multiple Additive Regression Trees) classifier."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for DART (Dropouts meet Multiple Additive Regression Trees) classifier.

## For Beginners

DART is gradient boosting with dropout - it randomly "forgets" some trees
during training to prevent overfitting. This is similar to dropout in neural networks.

Key options:

- DropoutRate: Fraction of trees to drop each iteration (higher = more regularization)
- DropoutType: How to select which trees to drop
- NormalizationType: How to scale predictions after dropout

## How It Works

DART applies dropout regularization to gradient boosting classification. During each iteration,
a random subset of existing trees is dropped, helping prevent overfitting.

## Properties

| Property | Summary |
|:-----|:--------|
| `DropoutRate` | Gets or sets the dropout rate - fraction of trees to drop each iteration. |
| `DropoutType` | Gets or sets the type of dropout selection. |
| `EarlyStoppingRounds` | Gets or sets early stopping rounds. |
| `LearningRate` | Gets or sets the learning rate (shrinkage factor). |
| `MaxDepth` | Gets or sets the maximum depth of trees. |
| `MaxFeatures` | Gets or sets the fraction of features to consider for splits. |
| `MinSamplesLeaf` | Gets or sets the minimum number of samples at a leaf node. |
| `MinSamplesSplit` | Gets or sets the minimum number of samples required to split. |
| `NormalizationType` | Gets or sets the normalization type after dropout. |
| `NumberOfIterations` | Gets or sets the number of boosting iterations. |
| `SplitCriterion` | Gets or sets the split criterion for the base trees. |
| `Verbose` | Gets or sets whether to print verbose output. |
| `VerboseEval` | Gets or sets how often to print progress. |

