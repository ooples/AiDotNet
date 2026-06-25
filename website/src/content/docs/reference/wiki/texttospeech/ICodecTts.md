---
title: "ICodecTts<T>"
description: "Interface for codec-based TTS models that use neural audio codecs with language model decoding."
section: "API Reference"
---

`Interfaces` · `AiDotNet.TextToSpeech.Interfaces`

Interface for codec-based TTS models that use neural audio codecs with language model decoding.

## How It Works

Codec-based TTS models use a neural audio codec (e.g., EnCodec, SoundStream, DAC) to represent
audio as discrete tokens, then use a language model to predict those tokens from text:
Text -> [LM] -> Audio Tokens -> [Codec Decoder] -> Waveform.
Architectures include:

- AR + NAR: VALL-E (autoregressive first codebook + non-autoregressive rest)
- Flow matching: CosyVoice, Voicebox (conditional flow matching on codec tokens)
- LLM-based: Fish Speech, Llasa (fine-tuned LLM predicts codec tokens)
- Parallel: SoundStorm (MaskGIT-style parallel decoding)

## Properties

| Property | Summary |
|:-----|:--------|
| `CodebookSize` | Gets the codebook vocabulary size. |
| `CodecFrameRate` | Gets the codec frame rate in Hz (tokens per second of audio). |
| `NumCodebooks` | Gets the number of residual vector quantization codebooks. |

## Methods

| Method | Summary |
|:-----|:--------|
| `DecodeFromTokens(Tensor<>)` | Decodes codec tokens back to audio waveform. |
| `EncodeToTokens(Tensor<>)` | Encodes audio into discrete codec tokens. |

