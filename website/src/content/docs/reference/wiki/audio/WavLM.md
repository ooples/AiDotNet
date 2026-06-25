---
title: "WavLM<T>"
description: "WavLM self-supervised speech representation model from Microsoft."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Audio.Foundations`

WavLM self-supervised speech representation model from Microsoft.

## For Beginners

WavLM improves on HuBERT by also learning to handle noisy audio.
During training, it was given audio with added noise and had to understand it anyway. This
makes it especially good at tasks in real-world noisy conditions, like telling speakers
apart in a meeting or understanding speech in a crowded room.

**Usage:**

## How It Works

WavLM (Chen et al., 2022, Microsoft) extends HuBERT with gated relative position bias
and denoising pre-training objectives. It achieves state-of-the-art on the SUPERB benchmark
and is particularly strong for speaker-related tasks due to its denoising training.

## Properties

| Property | Summary |
|:-----|:--------|
| `EmbeddingDimension` |  |
| `NumLayers` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ExtractEmbeddings(Tensor<>)` |  |
| `ExtractEmbeddingsAsync(Tensor<>,CancellationToken)` |  |
| `ExtractLayerFeatures(Tensor<>,Int32)` |  |
| `ExtractWeightedFeatures(Tensor<>,[])` |  |

