---
title: "ILabelMixingAugmentation<T, TData>"
description: "Interface for augmentations that modify labels (e.g., Mixup, CutMix)."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Augmentation`

Interface for augmentations that modify labels (e.g., Mixup, CutMix).

## Properties

| Property | Summary |
|:-----|:--------|
| `LastMixingLambda` | Gets the mixing lambda value from the last application. |

## Events

| Event | Summary |
|:-----|:--------|
| `OnLabelMixing` | Event raised when labels need to be mixed due to data mixing augmentation. |

