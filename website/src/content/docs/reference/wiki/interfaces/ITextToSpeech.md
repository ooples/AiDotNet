---
title: "ITextToSpeech<T>"
description: "Interface for text-to-speech (TTS) models that synthesize spoken audio from text."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Interface for text-to-speech (TTS) models that synthesize spoken audio from text.

## For Beginners

TTS is like having a computer read text out loud to you.

How TTS works:

1. Text is analyzed for pronunciation, emphasis, and pacing
2. The model generates audio features (mel-spectrograms)
3. A vocoder converts features to waveform audio

Common use cases:

- Accessibility (screen readers for visually impaired)
- Voice assistants and chatbots
- Audiobook and podcast generation
- Language learning applications

Key features:

- Voice cloning: Make it sound like a specific person
- Emotion control: Express happiness, sadness, excitement
- Speed control: Speak faster or slower

## How It Works

Text-to-speech models convert written text into natural-sounding spoken audio.
Modern TTS systems use neural networks to produce high-quality, expressive speech
that can sound nearly indistinguishable from human speakers.

This interface extends `IFullModel` for Tensor-based audio processing.

## Properties

| Property | Summary |
|:-----|:--------|
| `AvailableVoices` | Gets the list of available built-in voices. |
| `IsOnnxMode` | Gets whether this model is running in ONNX inference mode. |
| `SampleRate` | Gets the sample rate of generated audio. |
| `SupportsEmotionControl` | Gets whether this model supports emotional expression control. |
| `SupportsStreaming` | Gets whether this model supports streaming audio generation. |
| `SupportsVoiceCloning` | Gets whether this model supports voice cloning from reference audio. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ExtractSpeakerEmbedding(Tensor<>)` | Extracts speaker embedding from reference audio for voice cloning. |
| `StartStreamingSession(String,Double)` | Starts a streaming synthesis session for incremental audio generation. |
| `Synthesize(String,String,Double,Double)` | Synthesizes speech from text. |
| `SynthesizeAsync(String,String,Double,Double,CancellationToken)` | Synthesizes speech from text asynchronously. |
| `SynthesizeWithEmotion(String,String,Double,String,Double)` | Synthesizes speech with emotional expression. |
| `SynthesizeWithVoiceCloning(String,Tensor<>,Double,Double)` | Synthesizes speech using a cloned voice from reference audio. |

