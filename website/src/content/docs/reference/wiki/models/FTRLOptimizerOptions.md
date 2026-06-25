---
title: "FTRLOptimizerOptions<T, TInput, TOutput>"
description: "Configuration options for the Follow-The-Regularized-Leader (FTRL) optimizer, an advanced gradient-based optimization algorithm particularly effective for sparse datasets and online learning."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for the Follow-The-Regularized-Leader (FTRL) optimizer, an advanced gradient-based
optimization algorithm particularly effective for sparse datasets and online learning.

## For Beginners

Think of FTRL as a smart learning algorithm that adjusts how quickly your
model learns based on past experience. Unlike simpler optimizers that use the same learning approach
for all features, FTRL can learn different features at different rates. This makes it especially good
for data where many inputs might be zero or missing (called "sparse data"), like text analysis where
most words don't appear in most documents. FTRL was developed by Google and has been particularly
successful for online advertising and recommendation systems where models need to update continuously
as new data arrives.

## How It Works

FTRL (Follow-The-Regularized-Leader) is an optimization algorithm developed by Google that combines
the benefits of Adaptive Gradient (AdaGrad) for sparse data and Regularized Dual Averaging (RDA) for
better regularization. It's particularly effective for large-scale linear models and online learning
scenarios where data arrives sequentially.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FTRLOptimizerOptions` | Initializes a new instance of the FTRLOptimizerOptions class with appropriate defaults. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Alpha` | Gets or sets the alpha parameter, which controls the learning rate. |
| `BatchSize` | Gets or sets the batch size for mini-batch gradient descent. |
| `Beta` | Gets or sets the beta parameter, which helps prevent too large updates for infrequent features. |
| `Lambda1` | Gets or sets the L1 regularization strength, which encourages sparsity in the model. |
| `Lambda2` | Gets or sets the L2 regularization strength, which prevents any single feature from having too much influence. |
| `LearningRateDecreaseFactor` | Gets or sets the factor by which to decrease the learning rate when progress stalls or errors increase. |
| `LearningRateIncreaseFactor` | Gets or sets the factor by which to increase the learning rate when progress is good. |
| `MaxLearningRate` | Gets or sets the maximum learning rate allowed during training. |

