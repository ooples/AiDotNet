---
title: "ExtraTreesClassifierOptions<T>"
description: "Configuration options for Extra Trees (Extremely Randomized Trees) classifier."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for Extra Trees (Extremely Randomized Trees) classifier.

## For Beginners

Extra Trees is a "more random" version of Random Forest!

The key differences from Random Forest:

1. Random Forest finds the BEST split among random features
2. Extra Trees picks RANDOM splits for random features

This extra randomness:

- Makes training even faster
- Often generalizes better (less overfitting)
- Creates more diverse trees

When to use Extra Trees:

- When Random Forest is overfitting
- When you need faster training
- As an alternative to try alongside Random Forest

## How It Works

Extra Trees is similar to Random Forest but with even more randomization.
Instead of finding the best split, it selects splits at random, which can
lead to better generalization and faster training.

## Properties

| Property | Summary |
|:-----|:--------|
| `Bootstrap` | Gets or sets whether to bootstrap samples. |
| `Criterion` | Gets or sets the split criterion. |
| `MaxDepth` | Gets or sets the maximum depth of each tree. |
| `MaxFeatures` | Gets or sets the maximum number of features to consider. |
| `MinImpurityDecrease` | Gets or sets the minimum impurity decrease for splitting. |
| `MinSamplesLeaf` | Gets or sets the minimum samples required at a leaf. |
| `MinSamplesSplit` | Gets or sets the minimum samples required to split. |
| `NEstimators` | Gets or sets the number of trees in the forest. |

