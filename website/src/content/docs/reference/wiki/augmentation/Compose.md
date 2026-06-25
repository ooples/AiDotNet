---
title: "Compose<T, TData>"
description: "Applies multiple augmentations sequentially."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Augmentation`

Applies multiple augmentations sequentially.

## For Beginners

Think of this like a recipe with multiple steps.
First flip the image, then adjust brightness, then add noise. Each step
transforms the result of the previous step.

## How It Works

Compose chains multiple augmentations together, applying them one after another.
Each augmentation sees the output of the previous one. This is the most common
way to build augmentation pipelines.

**Example usage:**

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `Compose(IAugmentation<,>[])` | Creates a new composition of augmentations. |
| `Compose(IEnumerable<IAugmentation<,>>,Double)` | Creates a new composition of augmentations. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Augmentations` | Gets the list of augmentations in this composition. |
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
| `With(IAugmentation<,>)` | Creates a new composition with an additional augmentation appended. |

