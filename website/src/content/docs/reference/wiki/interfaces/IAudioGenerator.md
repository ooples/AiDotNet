---
title: "IAudioGenerator<T>"
description: "Interface for audio generation models that create audio from text descriptions or other conditions."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Interface for audio generation models that create audio from text descriptions or other conditions.

## For Beginners

Audio generation is like having an artist who can create
any sound you describe.

How audio generation works:

1. You provide a description ("A dog barking in a park")
2. The model generates audio features that match the description
3. The features are converted to playable audio

Types of audio generation:

- Text-to-Audio: "Thunder during a storm" creates thunder sounds
- Text-to-Music: "Upbeat jazz piano" creates music
- Audio Inpainting: Fill in missing parts of audio
- Audio Continuation: Extend existing audio naturally

Common use cases:

- Video game sound effects
- Film and media production
- Music composition assistance
- Podcast and content creation

## How It Works

Audio generation models create sounds, music, and audio effects from various inputs.
Unlike TTS which focuses on speech, audio generators can produce any type of sound
including music, environmental sounds, and sound effects.

This interface extends `IFullModel` for Tensor-based audio processing.

## Properties

| Property | Summary |
|:-----|:--------|
| `IsOnnxMode` | Gets whether this model is running in ONNX inference mode. |
| `MaxDurationSeconds` | Gets the maximum duration of audio that can be generated in seconds. |
| `SampleRate` | Gets the sample rate of generated audio. |
| `SupportsAudioContinuation` | Gets whether this model supports audio continuation. |
| `SupportsAudioInpainting` | Gets whether this model supports audio inpainting. |
| `SupportsTextToAudio` | Gets whether this model supports text-to-audio generation. |
| `SupportsTextToMusic` | Gets whether this model supports text-to-music generation. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ContinueAudio(Tensor<>,String,Double,Int32,Nullable<Int32>)` | Continues existing audio to extend it naturally. |
| `GenerateAudio(String,String,Double,Int32,Double,Nullable<Int32>)` | Generates audio from a text description. |
| `GenerateAudioAsync(String,String,Double,Int32,Double,Nullable<Int32>,CancellationToken)` | Generates audio from a text description asynchronously. |
| `GenerateMusic(String,String,Double,Int32,Double,Nullable<Int32>)` | Generates music from a text description. |
| `GetDefaultOptions` | Gets generation options for advanced control. |
| `InpaintAudio(Tensor<>,Tensor<>,String,Int32,Nullable<Int32>)` | Fills in missing or masked sections of audio. |

