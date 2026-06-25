---
title: "SpectrogramTransform<T>"
description: "Transforms raw audio waveform tensors into Mel spectrogram representations."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Transforms`

Transforms raw audio waveform tensors into Mel spectrogram representations.

## For Beginners

A spectrogram is a visual representation of audio frequencies over time.
Converting raw audio to a spectrogram is a common preprocessing step for audio ML models.

## How It Works

Wraps the existing `MelSpectrogram` to implement `ITransform`,
enabling composable use in data pipelines.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SpectrogramTransform(Int32,Int32,Int32,Int32,Boolean)` | Creates a new spectrogram transform. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Apply(Tensor<>)` | Applies the Mel spectrogram transform to a raw audio waveform tensor. |

