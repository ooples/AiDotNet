---
title: "AugmentationBase<T, TData>"
description: "Abstract base class for all augmentations providing common functionality."
section: "API Reference"
---

`Base Classes` · `AiDotNet.Augmentation`

Abstract base class for all augmentations providing common functionality.

## For Beginners

This is the foundation that all augmentations build upon.
It handles common tasks like deciding whether to apply the augmentation based on
probability, tracking parameters, and ensuring augmentations behave correctly
during training vs. inference.

## How It Works

AugmentationBase provides:

- Probability-based application control
- Training/inference mode awareness
- Parameter serialization
- Event hooks for monitoring

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AugmentationBase(Double)` | Initializes a new instance of the augmentation. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Engine` | Gets the hardware-accelerated computation engine for vectorized operations. |
| `IsEnabled` | Gets or sets whether this augmentation is currently enabled. |
| `IsTrainingOnly` | Gets whether this augmentation should only be applied during training. |
| `Name` | Gets the name of this augmentation. |
| `Probability` | Gets the probability of this augmentation being applied. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Apply(,AugmentationContext<>)` | Applies the augmentation to the input data. |
| `ApplyAugmentation(,AugmentationContext<>)` | Implement this method to perform the actual augmentation. |
| `GetParameters` | Gets the parameters of this augmentation. |
| `RaiseAugmentationApplied(AugmentationContext<>,Boolean)` | Raises the augmentation applied event. |

## Fields

| Field | Summary |
|:-----|:--------|
| `NumOps` | Numeric operations for type T. |

## Events

| Event | Summary |
|:-----|:--------|
| `OnAugmentationApplied` | Event raised when this augmentation is applied. |

