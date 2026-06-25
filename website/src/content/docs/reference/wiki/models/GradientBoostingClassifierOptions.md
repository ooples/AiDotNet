---
title: "GradientBoostingClassifierOptions<T>"
description: "Configuration options for Gradient Boosting classifier."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for Gradient Boosting classifier.

## For Beginners

Gradient Boosting is like iteratively fixing mistakes!

The process:

1. Start with a simple prediction (like the average)
2. Calculate how wrong we are (the residuals)
3. Train a tree to predict those errors
4. Add the tree's predictions (scaled down) to improve our model
5. Repeat many times

Key parameters to tune:

- n_estimators: More trees = potentially better but slower
- learning_rate: Lower values need more trees but often work better
- max_depth: Usually 3-8 works well (shallower than random forest)

Gradient Boosting often gives the best accuracy but:

- Takes longer to train
- More sensitive to hyperparameters
- More prone to overfitting if not tuned properly

## How It Works

Gradient Boosting builds an ensemble of trees where each tree corrects the errors
of the previous ones by fitting to the gradient of the loss function.

## Properties

| Property | Summary |
|:-----|:--------|
| `LearningRate` | Gets or sets the learning rate (shrinkage). |
| `Loss` | Gets or sets the loss function. |
| `MaxDepth` | Gets or sets the maximum depth of each tree. |
| `MaxFeatures` | Gets or sets the fraction of features used for each tree. |
| `MinImpurityDecrease` | Gets or sets the minimum impurity decrease for splitting. |
| `MinSamplesLeaf` | Gets or sets the minimum samples required at a leaf. |
| `MinSamplesSplit` | Gets or sets the minimum samples required to split. |
| `NEstimators` | Gets or sets the number of boosting stages. |
| `Subsample` | Gets or sets the fraction of samples used for each tree. |

