---
title: "SmoothedLossDifficultyEstimator<T, TInput, TOutput>"
description: "Loss-based difficulty estimator with moving average smoothing."
section: "API Reference"
---

`Models & Types` · `AiDotNet.CurriculumLearning.DifficultyEstimators`

Loss-based difficulty estimator with moving average smoothing.

## For Beginners

This variant uses a moving average of losses across
training epochs. This helps stabilize difficulty estimates and prevents sudden
changes in curriculum ordering due to random fluctuations.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SmoothedLossDifficultyEstimator(,ILossFunction<>,Boolean)` | Initializes a new instance of the `SmoothedLossDifficultyEstimator` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Name` | Gets the name of this estimator. |

## Methods

| Method | Summary |
|:-----|:--------|
| `EstimateDifficulties(IDataset<,,>,IFullModel<,,>)` | Estimates difficulty with exponential moving average smoothing. |
| `Reset` | Resets the estimator including smoothed values. |

