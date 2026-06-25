---
title: "IAugmentation<T, TData>"
description: "Base interface for all data augmentations across domains (image, audio, tabular)."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Augmentation`

Base interface for all data augmentations across domains (image, audio, tabular).

## For Beginners

Augmentation is like creating variations of your training data.
If you're training an image classifier, flipping images horizontally gives the model
more examples to learn from without collecting new data. This helps the model
generalize better to new, unseen data.

## How It Works

Augmentations are stochastic transformations applied during training to improve model
generalization. Unlike preprocessing transforms, augmentations:

- Produce different outputs for the same input (stochastic)
- Are typically disabled during inference (training-only)
- Have a probability of being applied
- Can be composed in pipelines

## Properties

| Property | Summary |
|:-----|:--------|
| `IsEnabled` | Gets whether this augmentation is currently enabled. |
| `IsTrainingOnly` | Gets whether this augmentation should only be applied during training. |
| `Name` | Gets the name of this augmentation for logging and debugging. |
| `Probability` | Gets the probability of this augmentation being applied (0.0 to 1.0). |

## Methods

| Method | Summary |
|:-----|:--------|
| `Apply(,AugmentationContext<>)` | Applies the augmentation to the input data. |
| `GetParameters` | Gets the parameters of this augmentation for serialization/logging. |

