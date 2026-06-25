---
title: "VALLE<T>"
description: "VALL-E zero-shot text-to-speech via neural codec language modeling."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Audio.Generation`

VALL-E zero-shot text-to-speech via neural codec language modeling.

## For Beginners

VALL-E can hear someone speak for 3 seconds and then generate new
speech in that person's voice. It works by converting speech into "audio words" (codec
tokens) and using a language model to predict what comes next. The AR model handles the
basic structure, and the NAR model adds the fine sound quality.

**Usage:**

## How It Works

VALL-E (Wang et al., 2023, Microsoft) treats TTS as a language modeling problem using
discrete audio codes from EnCodec. A 3-second enrollment recording suffices for zero-shot
voice synthesis. It uses an autoregressive (AR) model for the first codebook and a
non-autoregressive (NAR) model for the remaining codebook layers.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `VALLE(NeuralNetworkArchitecture<>,String,VALLEOptions)` | Creates a VALL-E model in ONNX inference mode. |
| `VALLE(NeuralNetworkArchitecture<>,VALLEOptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates a VALL-E model in native training mode. |

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
| `SynthesizeWithVoice(String,Tensor<>,Double)` | Synthesizes speech from text using a reference audio for voice cloning. |

