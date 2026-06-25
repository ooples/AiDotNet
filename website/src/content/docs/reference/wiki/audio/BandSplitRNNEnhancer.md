---
title: "BandSplitRNNEnhancer<T>"
description: "Band-Split RNN speech enhancement model (Luo and Yu, 2023)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Audio.Enhancement`

Band-Split RNN speech enhancement model (Luo and Yu, 2023).

## For Beginners

Imagine multiple specialized listeners, each focused on a different
pitch range (bass, midrange, treble). Each cleans up their range independently, then
they combine results. This divide-and-conquer approach works well because different
types of noise affect different frequency ranges differently.

**Usage:**

## How It Works

Band-Split RNN splits the spectrogram into non-overlapping frequency bands, processes
each band independently with a shared RNN, then fuses across bands. Originally designed
for music source separation, it also excels at speech enhancement by treating noise
as a source to separate.

## Properties

| Property | Summary |
|:-----|:--------|
| `EnhancementStrength` |  |
| `LatencySamples` |  |
| `NumChannels` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Enhance(Tensor<>)` |  |
| `EnhanceAsync(Tensor<>,CancellationToken)` |  |
| `EnhanceWithReference(Tensor<>,Tensor<>)` |  |
| `EstimateNoiseProfile(Tensor<>)` |  |
| `ProcessChunk(Tensor<>)` |  |

