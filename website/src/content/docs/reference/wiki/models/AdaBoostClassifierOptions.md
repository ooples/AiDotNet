---
title: "AdaBoostClassifierOptions<T>"
description: "Configuration options for AdaBoost classifier."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for AdaBoost classifier.

## For Beginners

AdaBoost is like a team of experts that learns from mistakes!

Imagine you have a series of simple decision makers:

1. The first one makes some mistakes
2. The second one focuses on fixing those mistakes
3. The third one focuses on fixing the remaining mistakes
4. And so on...

Each decision maker gets a "vote weight" based on how accurate it is.
The final prediction combines all their votes.

AdaBoost is great because:

- It automatically focuses on hard-to-classify samples
- It combines many simple rules into a complex decision boundary
- It's resistant to overfitting (in most cases)
- It provides a natural confidence measure

## How It Works

AdaBoost (Adaptive Boosting) is a meta-algorithm that combines multiple weak classifiers
into a strong classifier. Each subsequent classifier focuses more on the samples that
were misclassified by previous classifiers.

## Properties

| Property | Summary |
|:-----|:--------|
| `Algorithm` | Gets or sets the algorithm variant to use. |
| `LearningRate` | Gets or sets the learning rate (shrinkage). |
| `NEstimators` | Gets or sets the maximum number of estimators (weak learners). |

