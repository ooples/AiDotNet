---
title: "NGBoostClassifierOptions<T>"
description: "Configuration options for NGBoost (Natural Gradient Boosting) classifier models."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for NGBoost (Natural Gradient Boosting) classifier models.

## For Beginners

NGBoost is like regular gradient boosting for classification,
but it produces better-calibrated probability estimates. When NGBoost says there's
a 70% chance of class A, approximately 70% of similar predictions will actually
be class A.

This calibration is valuable because you can trust the probability estimates
for decision-making, not just the predicted class.

## How It Works

NGBoost is a probabilistic gradient boosting algorithm that outputs well-calibrated
class probabilities. It uses natural gradients (gradients preconditioned by the
Fisher Information Matrix) for more stable optimization.

## Properties

| Property | Summary |
|:-----|:--------|
| `ColumnSubsampleRatio` | Gets or sets whether to apply column subsampling. |
| `EarlyStoppingRounds` | Gets or sets the number of early stopping rounds. |
| `LearningRate` | Gets or sets the learning rate (shrinkage factor). |
| `MaxDepth` | Gets or sets the maximum depth of trees. |
| `MaxFeatures` | Gets or sets the fraction of features to consider for splits. |
| `MinSamplesLeaf` | Gets or sets the minimum number of samples required to be at a leaf node. |
| `MinSamplesSplit` | Gets or sets the minimum number of samples required to split an internal node. |
| `NumberOfIterations` | Gets or sets the number of boosting iterations (trees per class). |
| `SplitCriterion` | Gets or sets the split criterion for the base trees. |
| `SubsampleRatio` | Gets or sets the subsampling ratio for each iteration. |
| `UseNaturalGradient` | Gets or sets whether to use natural gradients. |
| `Verbose` | Gets or sets whether to verbose output during training. |
| `VerboseEval` | Gets or sets how often to print progress (every N iterations). |

