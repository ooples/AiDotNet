---
title: "JEN1Model<T>"
description: "JEN-1 model for high-fidelity text-to-music generation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Audio`

JEN-1 model for high-fidelity text-to-music generation.

## For Beginners

JEN-1 creates music from text descriptions:

How JEN-1 works:

1. Text is encoded by a FLAN-T5 text encoder
2. Audio is encoded by a 1D VAE into a compressed latent representation
3. A diffusion model denoises latents conditioned on text
4. The 1D VAE decodes latents back to 48kHz audio

Key characteristics:

- High-fidelity 48kHz audio output
- Text-to-music: Generate music from text descriptions
- Music continuation: Extend an existing music clip
- Music inpainting: Fill in missing parts of music
- Multi-task training: Autoregressive + non-autoregressive

Advantages:

- High audio quality (48kHz)
- Versatile (text-to-music, continuation, inpainting)
- Good musicality and coherence
- Efficient 1D latent representation

Limitations:

- Limited to ~10 second clips
- Generation takes several seconds
- Music quality varies with prompt complexity

## How It Works

JEN-1 is a universal high-fidelity music generation model that combines autoregressive
and non-autoregressive training in a multi-task framework. It generates music at 48kHz
in both text-to-music and music continuation modes.

Technical specifications:

- Architecture: 1D VAE + Latent Diffusion
- Audio encoder: 1D convolutional VAE, 128 latent channels
- Diffusion backbone: 1D U-Net with cross-attention
- Text encoder: FLAN-T5 Large (1024-dim embeddings)
- Sample rate: 48,000 Hz
- Duration: Up to 10 seconds
- Noise schedule: Linear beta, 1000 timesteps

Reference: Li et al., "JEN-1: Text-Guided Universal Music Generation with Omnidirectional Diffusion Models", 2023

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `JEN1Model(NeuralNetworkArchitecture<>,DiffusionModelOptions<>,INoiseScheduler<>,UNetNoisePredictor<>,AudioVAE<>,IConditioningModule<>,Nullable<Int32>)` | Initializes a new instance of JEN1Model with full customization support. |

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

