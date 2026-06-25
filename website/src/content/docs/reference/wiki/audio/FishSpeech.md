---
title: "FishSpeech<T>"
description: "Fish Speech open-source multilingual TTS with zero-shot voice cloning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Audio.Generation`

Fish Speech open-source multilingual TTS with zero-shot voice cloning.

## For Beginners

Fish Speech is a fast, open-source text-to-speech system. Give it
a few seconds of someone's voice and some text, and it speaks the text in that person's
voice. It works in many languages and is fast enough for live conversations.

**Usage:**

## How It Works

Fish Speech (Fish Audio, 2024) is an open-source multilingual TTS system that uses a
dual-AR architecture with grouped finite scalar quantization (GFSQ). It supports zero-shot
voice cloning from a few seconds of reference audio and generates natural speech in multiple
languages with very low latency suitable for real-time streaming.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FishSpeech(NeuralNetworkArchitecture<>,FishSpeechOptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates a Fish Speech model in native training mode. |
| `FishSpeech(NeuralNetworkArchitecture<>,String,FishSpeechOptions)` | Creates a Fish Speech model in ONNX inference mode. |

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
| `SynthesizeWithVoice(String,Tensor<>,Double)` | Synthesizes speech with zero-shot voice cloning from a reference audio. |
| `SynthesizeWithVoiceAsync(String,Tensor<>,Double,CancellationToken)` | Synthesizes speech with voice cloning asynchronously. |

