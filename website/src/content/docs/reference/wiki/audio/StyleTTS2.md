---
title: "StyleTTS2<T>"
description: "StyleTTS 2 text-to-speech model (Li et al., 2023)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Audio.TextToSpeech`

StyleTTS 2 text-to-speech model (Li et al., 2023).

## For Beginners

StyleTTS 2 is like having a professional voice actor:

1. You write the script (text)
2. You optionally provide a voice sample to clone (reference audio)
3. The model generates natural-sounding speech with realistic intonation

It's one of the most natural-sounding open-source TTS models available.

**Usage:**

## How It Works

StyleTTS 2 achieves human-level naturalness (MOS 4.16 on LJSpeech) by disentangling
speech into content and style, then using diffusion-based style generation. It supports
zero-shot voice cloning from a short reference clip and fine-grained prosody control.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `StyleTTS2(NeuralNetworkArchitecture<>,String,StyleTTS2Options)` | Creates a StyleTTS 2 model in ONNX inference mode. |
| `StyleTTS2(NeuralNetworkArchitecture<>,StyleTTS2Options,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates a StyleTTS 2 model in native training mode. |

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

