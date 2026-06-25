---
title: "LabelMixingAugmentationBase<T, TData>"
description: "Base class for label-mixing augmentations like Mixup and CutMix."
section: "API Reference"
---

`Base Classes` · `AiDotNet.Augmentation`

Base class for label-mixing augmentations like Mixup and CutMix.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LabelMixingAugmentationBase(Double,Double)` | Initializes a new label mixing augmentation. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Alpha` | Gets or sets the alpha parameter for Beta distribution sampling. |
| `LastMixingLambda` | Gets the mixing lambda from the last application. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetParameters` |  |
| `RaiseLabelMixing(LabelMixingEventArgs<>)` | Raises the label mixing event. |
| `SampleLambda(AugmentationContext<>)` | Samples a mixing lambda value from Beta(alpha, alpha) distribution. |

## Events

| Event | Summary |
|:-----|:--------|
| `OnLabelMixing` | Event raised when labels need to be mixed. |

