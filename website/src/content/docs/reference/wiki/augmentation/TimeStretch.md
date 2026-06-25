---
title: "TimeStretch<T>"
description: "Stretches or compresses audio in time without changing pitch."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Augmentation.Audio`

Stretches or compresses audio in time without changing pitch.

## For Beginners

Time stretching makes audio faster or slower without
changing the pitch - like how a slower speaker still has the same voice pitch.
This is different from simply playing at a different speed.

## How It Works

**When to use:**

- Speech recognition to handle different speaking speeds
- Music tempo adjustment
- Synchronizing audio with video

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TimeStretch(Double,Double,Double,Int32)` | Creates a new time stretch augmentation. |

## Properties

| Property | Summary |
|:-----|:--------|
| `MaxRate` | Gets the maximum time stretch factor. |
| `MinRate` | Gets the minimum time stretch factor. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyAugmentation(Tensor<>,AugmentationContext<>)` |  |
| `GetParameters` |  |

