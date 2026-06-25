---
title: "CodecTtsOptions"
description: "Base configuration options for codec-based TTS models."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.TextToSpeech.CodecBased`

Base configuration options for codec-based TTS models.

## For Beginners

These options configure the CodecTts model. Default values follow the original paper settings.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CodecTtsOptions` | Initializes a new instance with default values. |
| `CodecTtsOptions(CodecTtsOptions)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `CodebookSize` | Gets or sets the codebook vocabulary size. |
| `CodecFrameRate` | Gets or sets the codec frame rate in Hz. |
| `LLMDim` | Gets or sets the LLM hidden dimension. |
| `LanguageModelName` | Gets or sets the name of the underlying language model (e.g., "LLaMA", "GPT-2"). |
| `MaxCodecFrames` | Gets or sets the maximum generation length in codec frames. |
| `NumCodebooks` | Gets or sets the number of RVQ codebooks. |
| `NumLLMLayers` | Gets or sets the number of LLM decoder layers. |
| `SpeakerEmbeddingDim` | Gets or sets the speaker embedding dimension (for multi-speaker or cloning). |
| `TextEncoderDim` | Gets or sets the text encoder dimension. |

