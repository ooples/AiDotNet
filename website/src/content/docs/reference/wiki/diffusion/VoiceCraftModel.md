---
title: "VoiceCraftModel<T>"
description: "VoiceCraft model for zero-shot speech editing and text-to-speech with neural codec language modeling."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Audio`

VoiceCraft model for zero-shot speech editing and text-to-speech with neural codec language modeling.

## For Beginners

VoiceCraft edits and generates speech realistically from any voice.

How VoiceCraft works:

1. Reference audio is encoded into EnCodec tokens (multi-codebook representation)
2. Target text is encoded by Whisper into 768-dim conditioning features
3. For editing: tokens at the edit region are masked; for TTS: continuation tokens are masked
4. Causal masked transformer predicts tokens at masked positions via infilling
5. Delay pattern allows parallel prediction across codebook levels
6. EnCodec decoder reconstructs 16 kHz waveform from predicted tokens

Key characteristics:

- Zero-shot: works with any voice from a 3-second reference sample
- Speech editing: modify specific words while preserving voice and prosody
- TTS: generate speech from text matching the reference voice
- EnCodec token-based neural codec language model
- Delay pattern for efficient multi-codebook generation
- Open-source (MIT license)

When to use VoiceCraft:

- Zero-shot speech editing (correcting words in recordings)
- Zero-shot text-to-speech with voice cloning
- Audio post-production and correction
- Personalized speech synthesis from short samples

Limitations:

- 16 kHz output (lower than 24/44.1 kHz models)
- Requires aligned transcript for editing mode
- Quality depends on reference audio quality
- Longer edits may drift from reference prosody

## How It Works

VoiceCraft uses a token-infilling approach with a neural codec language model to achieve
high-quality zero-shot speech editing and text-to-speech synthesis. It can modify specific
words in existing audio or generate new speech matching a reference voice, all without
fine-tuning on the target speaker.

Architecture components:

- Causal masked transformer (2048 hidden, 16 layers, 16 heads) for token infilling
- EnCodec neural audio codec for tokenization and reconstruction
- Whisper-based text encoder for 768-dim conditioning
- Delay pattern for multi-codebook parallel prediction
- Token masking strategy for speech editing

Technical specifications:

- Architecture: Causal masked transformer with token infilling
- Hidden dimension: 2048
- Transformer layers: 16
- Attention heads: 16
- Text encoder: Whisper-based (768-dim)
- Audio codec: EnCodec (multi-codebook RVQ)
- Sample rate: 16,000 Hz
- Default duration: 20 seconds
- Mel channels: 80
- Open-source: Yes (MIT license)

Reference: Peng et al., "VoiceCraft: Zero-Shot Speech Editing and Text-to-Speech in the Wild", ACL 2024

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

