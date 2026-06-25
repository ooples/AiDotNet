---
title: "Pengi<T>"
description: "Pengi audio language model that frames all audio tasks as text-generation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Audio.Multimodal`

Pengi audio language model that frames all audio tasks as text-generation.

## For Beginners

Pengi treats all audio understanding as a conversation. Instead of
having separate models for "what sound is this?" and "describe this audio", Pengi uses one
model that can answer any question about audio by generating text responses. Think of it
as a chat assistant that can hear audio.

**Usage:**

## How It Works

Pengi (Deshmukh et al., 2023, Microsoft) is an audio language model that frames all audio
tasks as text-generation tasks. It uses a frozen Audio Spectrogram Transformer (AST) encoder
paired with a pre-trained language model, enabling open-ended audio reasoning, captioning,
and question answering without task-specific classification heads.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `Pengi(NeuralNetworkArchitecture<>,PengiOptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates a Pengi model in native training mode. |
| `Pengi(NeuralNetworkArchitecture<>,String,PengiOptions)` | Creates a Pengi model in ONNX inference mode. |

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

