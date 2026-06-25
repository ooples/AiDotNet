---
title: "IAudioDiffusionModel<T>"
description: "Interface for audio diffusion models that generate sound and music."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Interface for audio diffusion models that generate sound and music.

## For Beginners

Audio diffusion models work similarly to image diffusion,
but instead of generating pictures, they create sounds.

How audio diffusion works:

1. Audio is converted to a spectrogram (visual representation of sound)
2. Diffusion happens on this spectrogram (just like image diffusion)
3. The spectrogram is converted back to audio

Types of audio generation:

- Text-to-Audio: "A dog barking in a park" → audio clip
- Text-to-Music: "Upbeat jazz piano" → music track
- Text-to-Speech: Text → spoken voice
- Audio-to-Audio: Transform existing audio (voice conversion, style transfer)

Key challenges:

- Temporal coherence (sounds must flow naturally)
- Frequency relationships (harmonics, rhythm)
- Long-range dependencies (verse-chorus structure in music)

## How It Works

Audio diffusion models apply diffusion processes to generate audio content,
including music, speech, sound effects, and more. They typically operate on
audio spectrograms or mel-spectrograms in latent space.

This interface extends `IDiffusionModel` with audio-specific operations.

## Properties

| Property | Summary |
|:-----|:--------|
| `DefaultDurationSeconds` | Gets the default duration of generated audio in seconds. |
| `MelChannels` | Gets the number of mel spectrogram channels used. |
| `SampleRate` | Gets the sample rate of generated audio. |
| `SupportsAudioToAudio` | Gets whether this model supports audio-to-audio transformation. |
| `SupportsTextToAudio` | Gets whether this model supports text-to-audio generation. |
| `SupportsTextToMusic` | Gets whether this model supports text-to-music generation. |
| `SupportsTextToSpeech` | Gets whether this model supports text-to-speech generation. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AudioToAudio(Tensor<>,String,String,Double,Int32,Double,Nullable<Int32>)` | Transforms existing audio based on a text prompt. |
| `ContinueAudio(Tensor<>,String,Double,Int32,Nullable<Int32>)` | Continues/extends audio from a given clip. |
| `ExtractSpeakerEmbedding(Tensor<>)` | Gets speaker embeddings from a reference audio clip (for voice cloning). |
| `GenerateFromText(String,String,Nullable<Double>,Int32,Double,Nullable<Int32>)` | Generates audio from a text description. |
| `GenerateMusic(String,String,Nullable<Double>,Int32,Double,Nullable<Int32>)` | Generates music from a text description. |
| `MelSpectrogramToWaveform(Tensor<>)` | Converts mel spectrogram back to audio waveform. |
| `TextToSpeech(String,Tensor<>,Double,Int32,Nullable<Int32>)` | Synthesizes speech from text (text-to-speech). |
| `WaveformToMelSpectrogram(Tensor<>)` | Converts audio waveform to mel spectrogram. |

