---
title: "MiniBatchGradientDescentOptions<T, TInput, TOutput>"
description: "Configuration options for Mini-Batch Gradient Descent, an optimization algorithm that updates model parameters using the average gradient computed from small random subsets of training data."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for Mini-Batch Gradient Descent, an optimization algorithm that
updates model parameters using the average gradient computed from small random subsets of training data.

## For Beginners

Mini-Batch Gradient Descent is a method for training machine learning models
that tries to find the best values for the model's internal settings (parameters).

Imagine you're trying to find the lowest point in a hilly landscape while blindfolded:

- Full Batch Gradient Descent: You survey the entire landscape before taking each step
- Stochastic Gradient Descent: You take a step based on checking just one random spot
- Mini-Batch Gradient Descent: You check a small random sample of spots before each step

This middle-ground approach is popular because:

- It's faster than checking the entire landscape each time
- It's more stable than making decisions based on just one spot
- It works well with modern hardware that can efficiently process small batches

This class allows you to configure how this learning process works: how many examples to look at
in each batch, how long to train, and how the algorithm adjusts its step size over time.

## How It Works

Mini-Batch Gradient Descent is a variation of the gradient descent optimization algorithm that strikes a
balance between the efficiency of stochastic gradient descent and the stability of batch gradient descent.
It updates model parameters after processing small randomly-selected subsets (mini-batches) of the training
data, rather than processing individual samples (as in stochastic gradient descent) or the entire dataset
(as in batch gradient descent). This approach often converges faster than batch methods while providing
more stable updates than purely stochastic methods.

## Properties

| Property | Summary |
|:-----|:--------|
| `BatchSize` | Gets or sets the number of training examples used in each mini-batch. |
| `LearningRateDecreaseFactor` | Gets or sets the factor by which the learning rate is decreased when the loss is getting worse. |
| `LearningRateIncreaseFactor` | Gets or sets the factor by which the learning rate is increased when the loss is improving. |
| `MaxEpochs` | Gets or sets the maximum number of complete passes through the training dataset. |
| `MaxLearningRate` | Gets or sets the maximum allowed learning rate for the optimization process. |

