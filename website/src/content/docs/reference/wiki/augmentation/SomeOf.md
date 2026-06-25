---
title: "SomeOf<T, TData>"
description: "Randomly selects and applies N augmentations from a set."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Augmentation`

Randomly selects and applies N augmentations from a set.

## For Beginners

Think of this like drawing cards from a deck.
You pick 2 or 3 random transformations and apply them one after another.
This creates more variety in your training data.

## How It Works

SomeOf randomly chooses N augmentations from the provided set and applies them
in sequence. The number N can be fixed or randomly sampled from a range.
This provides more variety than OneOf while still limiting the total augmentation.

**Example usage:**

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SomeOf(Int32,IAugmentation<,>[])` | Creates a new SomeOf with a fixed number of augmentations. |
| `SomeOf(Int32,IEnumerable<IAugmentation<,>>,Double,Boolean)` | Creates a new SomeOf with a fixed number of augmentations. |
| `SomeOf(Int32,Int32,IEnumerable<IAugmentation<,>>,Double,Boolean)` | Creates a new SomeOf with a range of augmentations to apply. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Augmentations` | Gets the list of augmentations to choose from. |
| `IsEnabled` |  |
| `IsTrainingOnly` |  |
| `MaxN` | Gets the maximum number of augmentations to apply. |
| `MinN` | Gets the minimum number of augmentations to apply. |
| `Name` |  |
| `Probability` |  |
| `RandomizeOrder` | Gets whether the selected augmentations should be applied in random order. |
| `SupportsBoundingBoxes` |  |
| `SupportsKeypoints` |  |
| `SupportsMasks` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Apply(,AugmentationContext<>)` |  |
| `ApplyWithTargets(AugmentedSample<,>,AugmentationContext<>)` |  |
| `GetParameters` |  |

