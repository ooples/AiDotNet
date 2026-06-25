---
title: "AugmentationPipeline<T, TData>"
description: "Represents a pipeline of augmentations that are applied in sequence or composition."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Augmentation`

Represents a pipeline of augmentations that are applied in sequence or composition.

## For Beginners

Think of this as a recipe of transformations.
You might want to first flip an image, then rotate it, then adjust the colors.
This pipeline handles all of that automatically.

## How It Works

AugmentationPipeline allows you to compose multiple augmentations together.
Each augmentation in the pipeline is applied with its own probability,
and the order can be sequential, random, or one-of.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AugmentationPipeline(String)` | Creates a new augmentation pipeline with an optional name. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AugmentationCount` |  |
| `AugmentationNames` |  |
| `Augmentations` |  |
| `Name` |  |
| `Order` | Gets or sets the application order for augmentations. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Add(IAugmentation<,>)` | Adds an augmentation to the pipeline. |
| `AddRange(IEnumerable<IAugmentation<,>>)` | Adds multiple augmentations to the pipeline. |
| `Apply(,AugmentationContext<>)` |  |
| `GetConfiguration` |  |
| `OneOf(IAugmentation<,>[])` | Creates a sub-pipeline that applies one-of the specified augmentations. |
| `Shuffle(IAugmentation<,>[])` | Creates a sub-pipeline with random ordering. |

