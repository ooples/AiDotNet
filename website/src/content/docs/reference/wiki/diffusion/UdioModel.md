---
title: "UdioModel<T>"
description: "Udio/Suno architecture model for full-song music generation with structural awareness."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Audio`

Udio/Suno architecture model for full-song music generation with structural awareness.

## For Beginners

Udio/Suno generates complete songs from text prompts.

How Udio/Suno works:

1. Text prompt is encoded with genre-aware text encoder (1024-dim features)
2. Structural conditioning identifies verse/chorus/bridge sections
3. DiT backbone generates audio latents with musical structure awareness
4. Flow matching provides efficient denoising in fewer steps
5. High-fidelity audio VAE decodes latents to 44.1 kHz stereo audio
6. Post-processing applies mastering for production-quality output

Key characteristics:

- Full-song generation (2-5 minutes) with musical structure
- Verse/chorus/bridge structural conditioning
- Genre-aware generation across any musical style
- High-fidelity 44.1 kHz stereo output
- Optional vocal + instrumental separation
- Flow matching for efficient sampling

When to use Udio/Suno:

- Complete song generation from text descriptions
- Music production prototyping and ideation
- Background music generation for content
- Genre-specific music creation

Limitations:

- Commercial API service (not open-source)
- Generation time increases with song duration
- Limited fine-grained control over arrangement
- Vocal quality varies by genre and language

## How It Works

This model represents the Udio/Suno class of full-song music generation systems using
a DiT backbone with structural conditioning for verse/chorus/bridge awareness. It generates
complete songs at 44.1 kHz stereo with full musical structure from text prompts.

Architecture components:

- DiT backbone (2048 hidden, 32 layers, 16 heads) for music generation
- Structural conditioning for verse/chorus/bridge awareness
- Genre-aware text encoder (1024-dim) for style control
- High-fidelity audio VAE with 128 mel channels
- Flow matching scheduler for efficient inference
- Stereo output at 44.1 kHz sample rate

Technical specifications:

- Architecture: DiT with structural conditioning
- Hidden dimension: 2048
- Transformer layers: 32
- Attention heads: 16
- Context dimension: 1024
- Latent channels: 64
- Sample rate: 44,100 Hz (stereo)
- Default duration: 180 seconds (3 minutes)
- Maximum duration: 300 seconds (5 minutes)
- Mel channels: 128
- Scheduler: Flow matching

Reference: Conceptual representation of Udio/Suno-class music generation, 2024

## Properties

| Property | Summary |
|:-----|:--------|
| `Conditioner` |  |
| `LatentChannels` |  |
| `NoisePredictor` |  |
| `ParameterCount` |  |
| `SupportsAudioToAudio` |  |
| `SupportsTextToAudio` |  |
| `SupportsTextToMusic` |  |
| `SupportsTextToSpeech` |  |
| `VAE` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clone` |  |
| `DeepCopy` |  |
| `GetModelMetadata` |  |
| `GetParameters` |  |
| `SetParameters(Vector<>)` |  |

