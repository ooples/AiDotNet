---
title: "AudioNoise<T>"
description: "Adds background noise to audio data."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Augmentation.Audio`

Adds background noise to audio data.

## For Beginners

This augmentation adds random noise to audio,
simulating real-world recording conditions like background hum, ambient sounds,
or electronic interference. This helps models become robust to noisy inputs.

## How It Works

**SNR (Signal-to-Noise Ratio):**

- Higher SNR = less noise (cleaner audio)
- Lower SNR = more noise (noisier audio)
- 20 dB = barely audible noise
- 10 dB = noticeable noise
- 0 dB = signal and noise are equal

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AudioNoise(Double,Double,Double,Int32)` | Creates a new audio noise augmentation. |

## Properties

| Property | Summary |
|:-----|:--------|
| `MaxSnrDb` | Gets the maximum signal-to-noise ratio in dB. |
| `MinSnrDb` | Gets the minimum signal-to-noise ratio in dB. |
| `NoiseType` | Gets or sets the type of noise to add. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyAugmentation(Tensor<>,AugmentationContext<>)` |  |
| `GetParameters` |  |

