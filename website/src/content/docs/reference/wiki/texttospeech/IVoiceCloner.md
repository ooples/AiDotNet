---
title: "IVoiceCloner<T>"
description: "Interface for TTS models that support zero-shot or few-shot voice cloning from reference audio."
section: "API Reference"
---

`Interfaces` · `AiDotNet.TextToSpeech.Interfaces`

Interface for TTS models that support zero-shot or few-shot voice cloning from reference audio.

## How It Works

Voice cloning models can replicate a target speaker's voice from a short reference audio sample:

- Zero-shot: VALL-E, CosyVoice (3-10 seconds of reference audio)
- Few-shot: GPT-SoVITS, XTTS-v2 (minutes of reference audio)
- Instant: OpenVoice (separate tone color converter)

## Properties

| Property | Summary |
|:-----|:--------|
| `MinReferenceDuration` | Gets the minimum reference audio duration in seconds required for cloning. |
| `SpeakerEmbeddingDim` | Gets the speaker embedding dimensionality. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ExtractSpeakerEmbedding(Tensor<>)` | Extracts a speaker embedding from reference audio. |
| `SynthesizeWithVoice(String,Tensor<>)` | Synthesizes speech in the voice of a reference speaker. |

