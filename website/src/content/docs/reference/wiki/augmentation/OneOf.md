---
title: "OneOf<T, TData>"
description: "Randomly selects and applies exactly one augmentation from a set."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Augmentation`

Randomly selects and applies exactly one augmentation from a set.

## For Beginners

Think of this like flipping a coin to decide
which transformation to apply. You might flip the image OR rotate it,
but not both at the same time.

## How It Works

OneOf randomly chooses one augmentation from the provided set and applies it.
This is useful when you want variety but don't want to apply multiple similar
augmentations that might compound their effects too strongly.

**Example usage:**

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `OneOf(IAugmentation<,>[])` | Creates a new OneOf with uniform weights. |
| `OneOf(IEnumerable<IAugmentation<,>>,Double)` | Creates a new OneOf with uniform weights. |
| `OneOf(IEnumerable<ValueTuple<IAugmentation<,>,Double>>,Double)` | Creates a new OneOf with specified weights. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Augmentations` | Gets the list of augmentations to choose from. |
| `IsEnabled` |  |
| `IsTrainingOnly` |  |
| `Name` |  |
| `Probability` |  |
| `SupportsBoundingBoxes` |  |
| `SupportsKeypoints` |  |
| `SupportsMasks` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Apply(,AugmentationContext<>)` |  |
| `ApplyWithTargets(AugmentedSample<,>,AugmentationContext<>)` |  |
| `GetParameters` |  |

