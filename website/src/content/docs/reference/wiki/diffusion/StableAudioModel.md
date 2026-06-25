---
title: "StableAudioModel<T>"
description: "Stable Audio Open model — DiT-based latent diffusion for long-form audio generation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Audio`

Stable Audio Open model — DiT-based latent diffusion for long-form audio generation.

## For Beginners

Stable Audio creates music and sound effects from text:

How Stable Audio works:

1. Text is encoded using a CLAP text encoder for audio-aware embeddings
2. A DiT (Diffusion Transformer) denoises latents conditioned on text and timing
3. Timing embeddings (start_s, total_s) control where and how long to generate
4. An autoencoder decodes the latents back to 44.1 kHz stereo audio

Key characteristics:

- DiT backbone with timing conditioning (start_s, total_s)
- 44.1 kHz sample rate for high-fidelity audio
- Autoencoder-based latent space
- Up to 47 seconds of stereo audio
- Supports both music and sound effect generation

Advantages:

- High audio quality at 44.1 kHz
- Variable-length generation via timing conditioning
- Strong prompt adherence through CLAP embeddings
- Open-source weights and architecture

Limitations:

- Maximum duration of 47 seconds
- No text-to-speech capability
- Requires significant compute for real-time generation

Reference: Evans et al., "Stable Audio Open", Stability AI, 2024

## How It Works

Stable Audio uses a DiT operating on audio latents with timing conditioning
for variable-length, high-quality audio generation at 44.1 kHz.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `StableAudioModel(NeuralNetworkArchitecture<>,DiffusionModelOptions<>,INoiseScheduler<>,DiTNoisePredictor<>,AudioVAE<>,IConditioningModule<>)` | Initializes a new instance of StableAudioModel with full customization support. |

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

