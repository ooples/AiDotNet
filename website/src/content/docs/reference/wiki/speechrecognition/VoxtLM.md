---
title: "VoxtLM<T>"
description: "VoxtLM: decoder-only model for speech-text joint training"
section: "API Reference"
---

`Models & Types` · `AiDotNet.SpeechRecognition.Specialized`

VoxtLM: decoder-only model for speech-text joint training

## For Beginners

VoxtLM uses a single decoder-only Transformer for both ASR and TTS by representing speech and text as interleaved token sequences. Speech is tokenized using a neural codec (EnCodec) into discrete tokens at multiple quantization levels. The model i...

## How It Works

**References:**

- Paper: "VoxtLM: Unified Decoder-Only Models for Consolidating Speech Recognition/Synthesis and Speech/Text Continuation" (Maiti et al., 2024)

VoxtLM uses a single decoder-only Transformer for both ASR and TTS by representing speech and text as interleaved token sequences. Speech is tokenized using a neural codec (EnCodec) into discrete tokens at multiple quantization levels. The model is trained to autoregressively predict the next token whether it's speech or text, unifying recognition and synthesis. For ASR, the model takes speech codec tokens as input and generates text tokens.

## Methods

| Method | Summary |
|:-----|:--------|
| `Transcribe(Tensor<>,String,Boolean)` | Transcribes audio using VoxtLM's unified decoder-only architecture. |

