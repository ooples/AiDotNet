---
title: "SoundStormModel<T>"
description: "SoundStorm model for parallel masked audio token generation with conformer architecture."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Audio`

SoundStorm model for parallel masked audio token generation with conformer architecture.

## For Beginners

SoundStorm generates speech extremely quickly using parallel decoding.

How SoundStorm works:

1. Semantic tokens from AudioLM condition the generation (1024-dim)
2. All SoundStream residual tokens start fully masked
3. Conformer predicts token probabilities for all masked positions simultaneously
4. Highest-confidence tokens are unmasked each iteration
5. Process repeats with decreasing mask ratio until all tokens are revealed
6. SoundStream decoder converts tokens to a 24 kHz waveform

Key characteristics:

- Parallel generation (not auto-regressive) — 100x speedup
- MaskGIT-style iterative confidence-based unmasking
- SoundStream codec with residual vector quantization
- Conformer backbone for strong audio sequence modeling
- Conditioned on AudioLM semantic tokens for content control

When to use SoundStorm:

- Real-time or near-real-time speech synthesis
- High-throughput audio generation pipelines
- When generation speed is the primary concern
- Dialogue systems requiring low latency

Limitations:

- Requires pre-computed semantic tokens (AudioLM dependency)
- Quality slightly below auto-regressive baselines
- Fixed codec resolution from SoundStream
- Less flexible for music generation

## How It Works

SoundStorm uses MaskGIT-style parallel decoding of SoundStream tokens with a conformer
backbone, generating all residual quantization levels simultaneously for 100x faster
audio synthesis compared to auto-regressive approaches.

Architecture components:

- Conformer backbone (1024 hidden, 12 layers, 16 heads) for masked token prediction
- MaskGIT-style iterative parallel decoding over SoundStream tokens
- SoundStream codec with residual vector quantization
- Conditioning from AudioLM semantic tokens (1024-dim)
- Multi-level confidence-based unmasking schedule

Technical specifications:

- Architecture: Conformer with MaskGIT parallel decoding
- Hidden dimension: 1024
- Conformer layers: 12
- Attention heads: 16
- Conditioning: 1024-dim AudioLM semantic tokens
- Audio codec: SoundStream with residual VQ
- Sample rate: 24,000 Hz
- Default duration: 30 seconds
- Mel channels: 80
- Speedup: ~100x vs auto-regressive

Reference: Borsos et al., "SoundStorm: Efficient Parallel Audio Generation", 2023

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

