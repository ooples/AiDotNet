---
title: "Qwen2Audio<T>"
description: "Qwen2-Audio multimodal audio-language model for audio understanding and reasoning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Audio.Multimodal`

Qwen2-Audio multimodal audio-language model for audio understanding and reasoning.

## For Beginners

Qwen2-Audio can "listen" to audio and answer questions about it.
Play it music and ask "What genre is this?", play it a conversation and ask "What
language are they speaking?", or play environmental sounds and ask "Describe this scene."

**Usage:**

## How It Works

Qwen2-Audio (Chu et al., 2024, Alibaba) uses a Whisper-style audio encoder with a
Qwen2 language model backbone, connected by a perceiver-style adapter. It supports
audio captioning, question answering, sound event detection, and audio reasoning.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `Qwen2Audio(NeuralNetworkArchitecture<>,Qwen2AudioOptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates a Qwen2-Audio model in native training mode. |
| `Qwen2Audio(NeuralNetworkArchitecture<>,String,Qwen2AudioOptions)` | Creates a Qwen2-Audio model in ONNX inference mode. |

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

