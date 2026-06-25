---
title: "ConjugateGradientOptimizerOptions<T, TInput, TOutput>"
description: "Configuration options for the Conjugate Gradient optimization algorithm, which is used to train machine learning models."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for the Conjugate Gradient optimization algorithm, which is used to train machine learning models.

## For Beginners

Think of training a machine learning model like finding the lowest point in a valley.
The Conjugate Gradient method is like a smart hiker who remembers previous paths they've taken and uses that
information to find shortcuts to the bottom. This is often faster than basic methods that only look at the
current slope. This class lets you control how this "smart hiker" behaves - how big steps they take, when they
decide they're close enough to the bottom, and how many attempts they'll make before giving up.

## How It Works

The Conjugate Gradient method is an advanced optimization algorithm that often converges faster than
standard gradient descent. It uses information from previous iterations to determine more efficient search directions,
making it particularly effective for problems with many parameters.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ConjugateGradientOptimizerOptions` | Initializes a new instance of the ConjugateGradientOptimizerOptions class with appropriate defaults. |

## Properties

| Property | Summary |
|:-----|:--------|
| `BatchSize` | Gets or sets the batch size for gradient computation. |
| `InitialLearningRate` | Gets or sets the initial learning rate, which controls the size of the first optimization steps. |
| `LearningRateDecreaseFactor` | Gets or sets the factor by which to decrease the learning rate when progress stalls or errors increase. |
| `LearningRateIncreaseFactor` | Gets or sets the factor by which to increase the learning rate when progress is good. |
| `MaxLearningRate` | Gets or sets the maximum learning rate, which prevents the steps from becoming too large. |
| `MinLearningRate` | Gets or sets the minimum learning rate, which prevents the steps from becoming too small. |

