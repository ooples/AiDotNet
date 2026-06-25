---
title: "E2TTS<T>"
description: "E2 TTS: fully non-autoregressive flow-matching TTS with character-level text input."
section: "API Reference"
---

`Models & Types` · `AiDotNet.TextToSpeech.FlowDiffusion`

E2 TTS: fully non-autoregressive flow-matching TTS with character-level text input.

## For Beginners

E2 TTS: fully non-autoregressive flow-matching TTS with character-level text input.. This model converts text input into speech audio output.

## How It Works

**References:**

- Paper: "Embarrassingly Easy Text-to-Speech" (Eskimez et al., 2024)

## Methods

| Method | Summary |
|:-----|:--------|
| `DecodeFromTokens(Tensor<>)` | Decodes codec tokens to audio. |
| `EncodeToTokens(Tensor<>)` | Encodes audio to codec tokens. |
| `Synthesize(String)` | Synthesizes speech. |

