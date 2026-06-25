---
title: "TimeShift<T>"
description: "Shifts audio forward or backward in time."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Augmentation.Audio`

Shifts audio forward or backward in time.

## For Beginners

Time shifting moves audio forward or backward,
like adding silence at the beginning or end. This simulates different
recording start times and helps models handle timing variations.

## How It Works

**Handling shifted samples:**

- Wrap: Samples that go off one end appear at the other (circular)
- Zero: Shifted areas are filled with silence

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TimeShift(Double,Double,Double,Int32)` | Creates a new time shift augmentation. |

## Properties

| Property | Summary |
|:-----|:--------|
| `MaxShiftFraction` | Gets the maximum shift as a fraction of total duration. |
| `MinShiftFraction` | Gets the minimum shift as a fraction of total duration. |
| `WrapAround` | Gets or sets whether to wrap shifted samples (true) or fill with zeros (false). |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyAugmentation(Tensor<>,AugmentationContext<>)` |  |
| `GetParameters` |  |

