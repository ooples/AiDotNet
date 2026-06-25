---
title: "BarkModel<T>"
description: "Bark model for transformer-based text-to-audio generation with multi-lingual speech, music, and sound effects."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Audio`

Bark model for transformer-based text-to-audio generation with multi-lingual speech, music, and sound effects.

## For Beginners

Bark generates realistic speech and sounds from text prompts.

How Bark works:

1. Text is tokenized and encoded via CLIP into 768-dim conditioning features
2. Semantic transformer converts text tokens to high-level semantic audio tokens
3. Coarse acoustic transformer maps semantic tokens to coarse EnCodec tokens
4. Fine acoustic transformer refines coarse tokens to full-resolution EnCodec tokens
5. EnCodec decoder converts audio tokens to a 24 kHz waveform
6. Speaker presets enable voice cloning from short reference audio

Key characteristics:

- Three-stage GPT-like auto-regressive generation
- Multi-lingual speech in 10+ languages
- Non-speech audio: laughter, music, sound effects, sighing
- Speaker cloning with voice presets
- EnCodec-based audio codec at 24 kHz
- Open-source (Suno AI, MIT license)

When to use Bark:

- Multi-lingual text-to-speech generation
- Expressive speech with emotions and non-verbal sounds
- Quick audio prototyping from text descriptions
- When diverse audio output types are needed

Limitations:

- Auto-regressive generation is slower than parallel methods
- Maximum duration limited by context window
- Speaker cloning quality depends on reference audio
- Less control over fine-grained prosody

## How It Works

Bark uses a GPT-like auto-regressive architecture with three transformer stages to generate
diverse audio content from text prompts, including speech in 10+ languages, music, laughter,
sighing, and other non-verbal sounds. Audio tokens are produced via an EnCodec codec.

Architecture components:

- Semantic transformer (GPT-like, 1024 hidden, 24 layers, 16 heads) for text-to-semantic tokens
- Coarse acoustic transformer for semantic-to-coarse audio tokens
- Fine acoustic transformer for coarse-to-fine audio token refinement
- CLIP text encoder for 768-dim conditioning
- EnCodec-based audio codec for token-to-waveform synthesis
- Speaker voice presets for zero-shot cloning

Technical specifications:

- Architecture: Three-stage GPT-like transformer
- Hidden dimension: 1024
- Transformer layers: 24
- Attention heads: 16
- Text encoder: CLIP (768-dim)
- Audio codec: EnCodec
- Sample rate: 24,000 Hz
- Default duration: 15 seconds
- Mel channels: 100
- Open-source: Yes (MIT license)

Reference: Suno AI, "Bark: Text-Prompted Generative Audio Model", 2023

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

