---
title: "AudioLM<T>"
description: "AudioLM hierarchical audio language model for high-quality audio generation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Audio.Generation`

AudioLM hierarchical audio language model for high-quality audio generation.

## For Beginners

AudioLM generates natural-sounding audio by "thinking" about it
at two levels: first the big-picture meaning (semantic tokens), then the fine sound
details (acoustic tokens). Think of it like writing a story: first an outline, then the
vivid details. This produces audio that is both coherent and high-fidelity.

**Usage:**

## How It Works

AudioLM (Borsos et al., 2023, Google) generates high-quality, coherent audio by
combining semantic tokens (from a self-supervised model like w2v-BERT) with acoustic
tokens (from a neural codec like SoundStream). A hierarchical language model generates
semantic tokens first for high-level structure, then acoustic tokens for fine detail.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AudioLM(NeuralNetworkArchitecture<>,AudioLMOptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates an AudioLM model in native training mode. |
| `AudioLM(NeuralNetworkArchitecture<>,String,AudioLMOptions)` | Creates an AudioLM model in ONNX inference mode. |

## Properties

| Property | Summary |
|:-----|:--------|
| `IsOnnxMode` |  |
| `MaxDurationSeconds` |  |
| `SupportsAudioContinuation` |  |
| `SupportsAudioInpainting` |  |
| `SupportsTextToAudio` |  |
| `SupportsTextToMusic` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ContinueAudio(Tensor<>,String,Double,Int32,Nullable<Int32>)` |  |
| `GenerateAudio(String,String,Double,Int32,Double,Nullable<Int32>)` |  |
| `GenerateAudioAsync(String,String,Double,Int32,Double,Nullable<Int32>,CancellationToken)` |  |
| `GenerateMusic(String,String,Double,Int32,Double,Nullable<Int32>)` |  |
| `GetDefaultOptions` |  |
| `InpaintAudio(Tensor<>,Tensor<>,String,Int32,Nullable<Int32>)` |  |

