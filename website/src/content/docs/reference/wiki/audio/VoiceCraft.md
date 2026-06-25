---
title: "VoiceCraft<T>"
description: "VoiceCraft neural codec language model for speech editing and zero-shot TTS."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Audio.Generation`

VoiceCraft neural codec language model for speech editing and zero-shot TTS.

## For Beginners

VoiceCraft can edit speech like you edit text - change specific words
in a recording while keeping the speaker's voice. It can also clone a voice from a few
seconds of audio and generate new speech. Think of it as "find and replace" for spoken words.

**Usage:**

## How It Works

VoiceCraft (Peng et al., 2024) uses a token rearrangement procedure with causal masking
that enables both editing existing speech (replacing/inserting words) and generating new
speech from a short prompt, achieving high naturalness and speaker similarity.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `VoiceCraft(NeuralNetworkArchitecture<>,String,VoiceCraftOptions)` | Creates a VoiceCraft model in ONNX inference mode. |
| `VoiceCraft(NeuralNetworkArchitecture<>,VoiceCraftOptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates a VoiceCraft model in native training mode. |

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
| `EditSpeech(Tensor<>,Int32,Int32,String)` | Edits speech by replacing a segment with new content guided by text. |
| `GenerateAudio(String,String,Double,Int32,Double,Nullable<Int32>)` |  |
| `GenerateAudioAsync(String,String,Double,Int32,Double,Nullable<Int32>,CancellationToken)` |  |
| `GenerateMusic(String,String,Double,Int32,Double,Nullable<Int32>)` |  |
| `GetDefaultOptions` |  |
| `InpaintAudio(Tensor<>,Tensor<>,String,Int32,Nullable<Int32>)` |  |

