---
title: "VolumeChange<T>"
description: "Randomly changes the volume (gain) of audio."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Augmentation.Audio`

Randomly changes the volume (gain) of audio.

## For Beginners

Volume change augmentation makes audio louder or quieter.
This helps models become robust to different recording volumes and microphone
distances.

## How It Works

**Gain in dB:**

- +6 dB ≈ 2x louder
- 0 dB = no change
- -6 dB ≈ 0.5x volume

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `VolumeChange(Double,Double,Double,Int32)` | Creates a new volume change augmentation. |

## Properties

| Property | Summary |
|:-----|:--------|
| `MaxGainDb` | Gets the maximum volume change in dB. |
| `MinGainDb` | Gets the minimum volume change in dB. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyAugmentation(Tensor<>,AugmentationContext<>)` |  |
| `GetParameters` |  |

