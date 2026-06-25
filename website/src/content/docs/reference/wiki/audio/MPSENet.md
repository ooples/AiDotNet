---
title: "MPSENet<T>"
description: "MP-SENet (Multi-Path Speech Enhancement Network) model (Lu et al., 2023)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Audio.Enhancement`

MP-SENet (Multi-Path Speech Enhancement Network) model (Lu et al., 2023).

## For Beginners

Sound has two components: loudness (magnitude) and timing (phase).
Most enhancers only fix loudness, leaving timing artifacts. MP-SENet fixes both at once
using two parallel paths that communicate, producing cleaner, more natural audio.

**Usage:**

## How It Works

MP-SENet predicts both magnitude and phase of the complex spectrogram using parallel
estimation paths with cross-domain fusion. It achieves PESQ 3.60 on VoiceBank+DEMAND,
surpassing prior single-channel speech enhancement methods.

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

