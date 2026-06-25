---
title: "ExplainableBoostingClassifierOptions<T>"
description: "Configuration options for Explainable Boosting Machine (EBM) classifier."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for Explainable Boosting Machine (EBM) classifier.

## For Beginners

EBM learns a separate "effect" for each feature, and the final
prediction is just the sum of these effects. This makes it easy to understand
exactly how each feature influences the prediction.

Key options:

- MaxBins: How finely to discretize continuous features (more bins = more detailed patterns)
- OuterBags: Number of boosting rounds over all features
- MaxInteractions: Number of feature pairs to consider for interactions

## How It Works

EBM is an interpretable machine learning algorithm that learns additive models with
optional pairwise interactions. It provides accuracy comparable to black-box models
while remaining fully interpretable.

## Properties

| Property | Summary |
|:-----|:--------|
| `EarlyStoppingRounds` | Gets or sets early stopping rounds. |
| `InnerBags` | Gets or sets the number of inner bags (boosting rounds per feature). |
| `L2Regularization` | Gets or sets the L2 regularization strength. |
| `LearningRate` | Gets or sets the learning rate for boosting updates. |
| `MaxBins` | Gets or sets the maximum number of bins for continuous features. |
| `MaxInteractionBins` | Gets or sets the maximum number of bins for interaction terms. |
| `MaxInteractionFeatures` | Gets or sets the maximum number of features to consider when searching for interactions. |
| `MaxInteractions` | Gets or sets the maximum number of pairwise interactions to detect. |
| `OuterBags` | Gets or sets the number of outer bags (boosting rounds over all features). |
| `Verbose` | Gets or sets whether to print verbose output. |
| `VerboseEval` | Gets or sets how often to print progress. |

