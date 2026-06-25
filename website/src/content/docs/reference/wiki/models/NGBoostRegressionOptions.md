---
title: "NGBoostRegressionOptions"
description: "Configuration options for NGBoost (Natural Gradient Boosting) regression models."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for NGBoost (Natural Gradient Boosting) regression models.

## For Beginners

NGBoost is like regular gradient boosting, but instead of predicting
a single number, it predicts a full probability distribution. This tells you not just
"what the prediction is" but also "how confident the model is."

For example, instead of predicting "house price = $300,000", NGBoost might predict
"house price is normally distributed with mean $300,000 and standard deviation $50,000."
This uncertainty information is valuable for decision-making.

## How It Works

NGBoost is a probabilistic gradient boosting algorithm that outputs full probability
distributions instead of point predictions. It uses natural gradients (gradients
preconditioned by the Fisher Information Matrix) for more stable optimization.

## Properties

| Property | Summary |
|:-----|:--------|
| `ColumnSubsampleRatio` | Gets or sets whether to apply column subsampling. |
| `DistributionType` | Gets or sets the type of distribution to fit. |
| `EarlyStoppingRounds` | Gets or sets the number of early stopping rounds. |
| `LearningRate` | Gets or sets the learning rate (shrinkage factor). |
| `MinSamplesSplit` | Gets or sets the minimum number of samples required to split an internal node. |
| `NumberOfIterations` | Gets or sets the number of boosting iterations (trees). |
| `ScoringRule` | Gets or sets the type of scoring rule used for optimization. |
| `SubsampleRatio` | Gets or sets the subsampling ratio for each iteration. |
| `UseNaturalGradient` | Gets or sets whether to use natural gradients. |
| `Verbose` | Gets or sets whether to verbose output during training. |
| `VerboseEval` | Gets or sets how often to print progress (every N iterations). |

