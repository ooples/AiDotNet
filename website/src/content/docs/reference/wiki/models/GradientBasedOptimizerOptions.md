---
title: "GradientBasedOptimizerOptions<T, TInput, TOutput>"
description: "Configuration options for gradient-based optimization algorithms."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for gradient-based optimization algorithms.

## For Beginners

Imagine you're in a hilly landscape and want to find the lowest point.
Gradient-based optimization is like always walking downhill in the steepest direction until you can't go any lower.
The "gradient" is simply the direction of the steepest slope at your current position.

## How It Works

Gradient-based optimizers are algorithms that find the minimum or maximum of a function
by following the direction of steepest descent or ascent (the gradient).

These algorithms are fundamental to training many machine learning models, including neural networks,
linear regression, and logistic regression.

This class inherits from `OptimizationAlgorithmOptions`, which means it includes all the
base configuration options for optimization algorithms plus any additional options specific to
gradient-based methods.

## Properties

| Property | Summary |
|:-----|:--------|
| `DataSampler` | Gets or sets the optional data sampler for advanced sampling strategies during batch creation. |
| `DropLastBatch` | Gets or sets whether to drop the last incomplete batch. |
| `EnableGradientClipping` | Gets or sets whether gradient clipping is enabled. |
| `GradientCache` | Gets or sets the gradient cache to use for storing and retrieving computed gradients. |
| `GradientClippingMethod` | Gets or sets the gradient clipping method to use. |
| `LearningRateScheduler` | Gets or sets the learning rate scheduler to use during training. |
| `LossFunction` | Gets or sets the loss function to use for evaluating model performance. |
| `LossFunctionExplicitlySet` | True when `LossFunction` was explicitly set by the caller; false while it still holds the default `MeanSquaredErrorLoss`. |
| `MaxGradientNorm` | Gets or sets the maximum gradient norm for norm-based clipping. |
| `MaxGradientValue` | Gets or sets the maximum gradient value for value-based clipping. |
| `RandomSeed` | Gets or sets the random seed for reproducibility. |
| `Regularization` | Gets or sets the regularization method to use for preventing overfitting. |
| `SchedulerStepMode` | Gets or sets when the learning rate scheduler should be stepped. |
| `ShuffleData` | Gets or sets whether to shuffle data at the beginning of each epoch. |

## Methods

| Method | Summary |
|:-----|:--------|
| `SetLossFunctionFromAutoSync(ILossFunction<>)` | Internal: assign `LossFunction` without flipping `LossFunctionExplicitlySet`. |

