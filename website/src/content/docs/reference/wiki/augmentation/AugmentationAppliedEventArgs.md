---
title: "AugmentationAppliedEventArgs<T>"
description: "Event arguments raised when an augmentation is applied."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Augmentation`

Event arguments raised when an augmentation is applied.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AugmentationAppliedEventArgs(String,IDictionary<String,Object>,Int32,Boolean)` | Creates a new augmentation applied event. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AugmentationName` | Gets the name of the augmentation that was applied. |
| `Parameters` | Gets the parameters used for this application. |
| `SampleIndex` | Gets the sample index within the batch. |
| `WasApplied` | Gets whether the augmentation was actually applied (vs. |

