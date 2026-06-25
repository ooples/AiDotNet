---
title: "MusicTaggingTransformer<T>"
description: "Music Tagging Transformer for multi-label music tag prediction (genre, mood, instrument, era)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Audio.MusicAnalysis`

Music Tagging Transformer for multi-label music tag prediction (genre, mood, instrument, era).

## For Beginners

This model listens to music and automatically tags it with descriptive
labels - like "rock", "upbeat", "guitar", "1980s", or "relaxing". It's the technology behind
automatic music categorization in streaming services like Spotify's genre detection.

**Usage:**

## How It Works

The Music Tagging Transformer (Won et al., 2021) uses a Transformer encoder on mel spectrogram
features to predict music tags (genre, mood, instrument, era). It achieves state-of-the-art
results on the MagnaTagATune and Million Song Dataset benchmarks.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MusicTaggingTransformer(NeuralNetworkArchitecture<>,MusicTaggingTransformerOptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates a Music Tagging Transformer in native training mode. |
| `MusicTaggingTransformer(NeuralNetworkArchitecture<>,String,MusicTaggingTransformerOptions)` | Creates a Music Tagging Transformer in ONNX inference mode. |

## Properties

| Property | Summary |
|:-----|:--------|
| `NumTags` | Gets the number of tags. |
| `TagLabels` | Gets the tag labels this model can predict. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetTagProbabilities(Tensor<>)` | Gets raw tag probabilities for all tags. |
| `GetTopKTags(Tensor<>,Int32)` | Gets the top-K tags by confidence. |
| `PredictTags(Tensor<>,Double)` | Predicts music tags for the given audio. |
| `PredictTagsAsync(Tensor<>,Double,CancellationToken)` | Predicts music tags asynchronously. |

