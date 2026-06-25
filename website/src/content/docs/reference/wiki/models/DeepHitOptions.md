---
title: "DeepHitOptions<T>"
description: "Configuration options for DeepHit survival analysis model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for DeepHit survival analysis model.

## For Beginners

While DeepSurv assumes that factors affect hazard rates
proportionally (like "smoking doubles your risk at all times"), DeepHit makes no
such assumption - it learns the actual probability of an event happening at each
specific time point.

DeepHit is particularly useful when:

- The proportional hazards assumption doesn't hold
- You have multiple competing risks (e.g., patient could die from disease OR treatment side effects)
- You want to predict exact survival probabilities at specific time horizons
- You need to handle complex, non-linear relationships in the data

For example: "What's the probability this patient survives past 1 year? 2 years? 5 years?"
DeepHit directly estimates these probabilities.

## How It Works

DeepHit is a deep learning approach to survival analysis that directly learns the
distribution of survival times without making assumptions like proportional hazards.
It can also handle competing risks (multiple possible failure types).

## Properties

| Property | Summary |
|:-----|:--------|
| `Activation` | Gets or sets the activation function for hidden layers. |
| `BatchSize` | Gets or sets the batch size for training. |
| `DropoutRate` | Gets or sets the dropout rate for regularization. |
| `EarlyStoppingPatience` | Gets or sets the patience for early stopping. |
| `Epochs` | Gets or sets the number of training epochs. |
| `EvaluationHorizons` | Gets or sets the time horizons of interest for evaluation. |
| `HiddenLayerSize` | Gets or sets the number of neurons in each hidden layer. |
| `L2Regularization` | Gets or sets the L2 regularization strength. |
| `LearningRate` | Gets or sets the learning rate for optimization. |
| `NumCauseLayers` | Gets or sets the number of hidden layers in each cause-specific sub-network. |
| `NumRisks` | Gets or sets the number of competing risks/causes. |
| `NumSharedLayers` | Gets or sets the number of hidden layers in the shared sub-network. |
| `NumTimeBins` | Gets or sets the number of discrete time bins for survival prediction. |
| `RankingSigma` | Gets or sets the sigma parameter for the ranking loss kernel. |
| `RankingWeight` | Gets or sets the weight for the ranking loss component. |
| `Seed` | Gets or sets the random seed for reproducibility. |
| `UseBatchNormalization` | Gets or sets whether to use batch normalization. |

