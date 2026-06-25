---
title: "CosyVoice2<T>"
description: "CosyVoice2 scalable streaming TTS model from Alibaba."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Audio.TextToSpeech`

CosyVoice2 scalable streaming TTS model from Alibaba.

## For Beginners

CosyVoice2 converts text into natural-sounding speech. What makes
it special is that it can clone anyone's voice from just a few seconds of audio, speak
in different languages, and add emotions. It's also fast enough for real-time applications
like voice assistants and audiobooks.

**Usage:**

## How It Works

CosyVoice2 (Du et al., 2024, Alibaba) uses a finite scalar quantization (FSQ) codec
with a flow-matching decoder to achieve natural-sounding speech with very low latency.
It supports zero-shot voice cloning from a few seconds of reference audio, cross-lingual
synthesis, and fine-grained emotion control.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CosyVoice2(NeuralNetworkArchitecture<>,CosyVoice2Options,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates a CosyVoice2 model in native training mode. |
| `CosyVoice2(NeuralNetworkArchitecture<>,String,CosyVoice2Options)` | Creates a CosyVoice2 model in ONNX inference mode. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AvailableVoices` |  |
| `SupportsEmotionControl` |  |
| `SupportsStreaming` |  |
| `SupportsVoiceCloning` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ExtractSpeakerEmbedding(Tensor<>)` |  |
| `StartStreamingSession(String,Double)` |  |
| `Synthesize(String,String,Double,Double)` |  |
| `SynthesizeAsync(String,String,Double,Double,CancellationToken)` |  |
| `SynthesizeWithEmotion(String,String,Double,String,Double)` |  |
| `SynthesizeWithVoiceCloning(String,Tensor<>,Double,Double)` |  |

