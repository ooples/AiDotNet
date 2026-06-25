---
title: "MusicFlamingo<T>"
description: "Music Flamingo multimodal music-language model for music understanding and reasoning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Audio.Multimodal`

Music Flamingo multimodal music-language model for music understanding and reasoning.

## For Beginners

Music Flamingo gives a language AI the ability to understand music.
You can play it a song and ask "What genre is this?" or "What instruments are playing?" and
it answers in natural language by combining its music listening with language understanding.

**Usage:**

## How It Works

Music Flamingo (2024) adapts the Flamingo architecture specifically for music understanding.
It uses a frozen music encoder (e.g., MERT or Jukebox features) with perceiver cross-attention
to enable a pre-trained LLM to reason about music: answering questions about genre, instruments,
mood, structure, and musical theory.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MusicFlamingo(NeuralNetworkArchitecture<>,MusicFlamingoOptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates a Music Flamingo model in native training mode. |
| `MusicFlamingo(NeuralNetworkArchitecture<>,String,MusicFlamingoOptions)` | Creates a Music Flamingo model in ONNX inference mode. |

## Properties

| Property | Summary |
|:-----|:--------|
| `MaxAudioDurationSeconds` |  |
| `MaxResponseTokens` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Caption(Tensor<>,Int32)` |  |
| `ExtractAudioEmbeddings(Tensor<>)` |  |
| `GetCapabilities` |  |
| `Understand(Tensor<>,String,Int32,Double)` |  |
| `UnderstandAsync(Tensor<>,String,Int32,Double,CancellationToken)` |  |

