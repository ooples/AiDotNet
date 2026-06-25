---
title: "AugmentationScheduler<T>"
description: "Schedules augmentation strength changes during training."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Augmentation.Image`

Schedules augmentation strength changes during training.

## Properties

| Property | Summary |
|:-----|:--------|
| `Augmenter` | Gets the underlying augmenter. |
| `CurrentStrength` | Gets the current augmentation strength factor [0, 1]. |

## Methods

| Method | Summary |
|:-----|:--------|
| `SetEpoch(Int32)` | Sets the current epoch. |
| `Step` | Updates the scheduler to the next epoch. |

