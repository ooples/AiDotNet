---
title: "TemporalFlip<T>"
description: "Reverses the order of frames in a video."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Augmentation.Video`

Reverses the order of frames in a video.

## For Beginners

Temporal flip plays the video backwards by reversing
the frame order. This can help models become invariant to action direction,
though it should be used carefully as some actions are not time-reversible.

## How It Works

**When to use:**

- Symmetric actions (walking back and forth, waving)
- Scene recognition where temporal direction doesn't matter
- Data augmentation for limited video datasets

**When NOT to use:**

- Actions with clear direction (falling down vs. jumping up)
- Tasks involving causality or temporal reasoning

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TemporalFlip(Double,Double)` | Creates a new temporal flip augmentation. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyAugmentation(ImageTensor<>[],AugmentationContext<>)` |  |

