---
title: "SALMONN<T>"
description: "SALMONN dual-encoder audio-language model for speech and audio understanding."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Audio.Multimodal`

SALMONN dual-encoder audio-language model for speech and audio understanding.

## For Beginners

SALMONN has two "ears": one for speech (Whisper) and one for
general sounds (BEATs). This means it can understand what people say AND non-speech
sounds. Ask it "What is the person saying?" and it transcribes speech. Ask "What sounds
are in the background?" and it identifies environmental audio.

**Usage:**

## How It Works

SALMONN (Tang et al., 2024, Tsinghua/ByteDance) uses dual audio encoders: a Whisper
speech encoder and a BEATs audio encoder, connected to a Vicuna LLM through a
window-level Q-Former adapter. This gives it strong capability for both speech and
general audio understanding tasks.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SALMONN(NeuralNetworkArchitecture<>,SALMONNOptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates a SALMONN model in native training mode. |
| `SALMONN(NeuralNetworkArchitecture<>,String,SALMONNOptions)` | Creates a SALMONN model in ONNX inference mode. |

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

