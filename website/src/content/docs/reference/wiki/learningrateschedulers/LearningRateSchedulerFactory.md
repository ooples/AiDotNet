---
title: "LearningRateSchedulerFactory"
description: "Factory for creating learning rate schedulers with common configurations."
section: "API Reference"
---

`Helpers & Utilities` · `AiDotNet.LearningRateSchedulers`

Factory for creating learning rate schedulers with common configurations.

## For Beginners

Instead of manually configuring schedulers with many parameters,
you can use this factory to create schedulers optimized for specific scenarios. For example,
CreateForTransformer() creates a scheduler tuned for transformer model training, with
warmup and linear decay that works well for attention-based models.

## How It Works

This factory provides convenient methods for creating pre-configured learning rate
schedulers for common use cases. It simplifies scheduler creation and provides
sensible defaults for various training scenarios.

## Methods

| Method | Summary |
|:-----|:--------|
| `Create(LearningRateSchedulerType,Double,Int32)` | Creates a scheduler based on type enum. |
| `CreateAdaptive(Double,Double,Int32,Double)` | Creates a learning rate scheduler that adapts based on validation loss. |
| `CreateForCNN(Double,Int32,Double)` | Creates a learning rate scheduler for typical CNN training. |
| `CreateForFineTuning(Double)` | Creates a learning rate scheduler for fine-tuning pre-trained models. |
| `CreateForLongTraining(Double,Int32,Double)` | Creates a learning rate scheduler for long training runs. |
| `CreateForSuperConvergence(Double,Int32,Double)` | Creates a learning rate scheduler for super-convergence training. |
| `CreateForTransformer(Double,Int32,Int32)` | Creates a learning rate scheduler for transformer training. |
| `CreateWithWarmRestarts(Double,Int32,Int32,Double)` | Creates a learning rate scheduler with warm restarts. |

