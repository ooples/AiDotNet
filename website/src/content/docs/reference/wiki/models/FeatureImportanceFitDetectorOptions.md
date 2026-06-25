---
title: "FeatureImportanceFitDetectorOptions"
description: "Configuration options for the Feature Importance Fit Detector, which analyzes how different input features contribute to a model's predictions and evaluates potential issues with model fit."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for the Feature Importance Fit Detector, which analyzes how different input features
contribute to a model's predictions and evaluates potential issues with model fit.

## For Beginners

Think of this as a tool that helps you understand which of your input data points
actually matter for making predictions. For example, if you're predicting house prices, this would tell you whether
square footage, number of bedrooms, or neighborhood has the biggest impact on price predictions. It also helps
identify potential problems with your model, like whether it's focusing too much on unimportant details or not
capturing important patterns. The options below let you adjust how sensitive this analysis should be.

## How It Works

Feature importance analysis helps identify which input variables have the strongest influence on model predictions.
This detector uses permutation importance (randomly shuffling feature values and measuring the impact on predictions)
to assess feature relevance and detect potential issues like overfitting, underfitting, or redundant features.

## Properties

| Property | Summary |
|:-----|:--------|
| `CorrelationThreshold` | Gets or sets the threshold for considering features as correlated. |
| `HighImportanceThreshold` | Gets or sets the threshold for considering feature importance as high. |
| `HighVarianceThreshold` | Gets or sets the threshold for considering importance variance as high. |
| `LowImportanceThreshold` | Gets or sets the threshold for considering feature importance as low. |
| `LowVarianceThreshold` | Gets or sets the threshold for considering importance variance as low. |
| `NumPermutations` | Gets or sets the number of permutations to perform for each feature when calculating importance. |
| `RandomSeed` | Gets or sets the random seed for feature permutation. |
| `UncorrelatedRatioThreshold` | Gets or sets the threshold for the ratio of uncorrelated feature pairs to consider features as mostly uncorrelated. |

