---
title: "Phi4Audio<T>"
description: "Phi-4 Audio: Microsoft's efficient speech-language model"
section: "API Reference"
---

`Models & Types` · `AiDotNet.SpeechRecognition.LLMIntegrated`

Phi-4 Audio: Microsoft's efficient speech-language model

## For Beginners

Phi-4 Audio extends the Phi-4 language model with speech understanding through a Mixture-of-LoRA adapter approach. A lightweight speech encoder processes audio and produces features that are injected into the Phi-4 model via specialized LoRA adapt...

## How It Works

**References:**

- Paper: "Phi-4-mini Technical Report: Compact yet Powerful Multimodal Language Models via Mixture-of-LoRAs" (Microsoft, 2025)

Phi-4 Audio extends the Phi-4 language model with speech understanding through a Mixture-of-LoRA adapter approach. A lightweight speech encoder processes audio and produces features that are injected into the Phi-4 model via specialized LoRA adapters. The mixture approach routes different speech feature types (prosody, phonetic, semantic) to specialized adapter experts. This enables efficient speech understanding without full model fine-tuning while maintaining Phi-4's strong language capabilities.

## Methods

| Method | Summary |
|:-----|:--------|
| `Transcribe(Tensor<>,String,Boolean)` | Transcribes audio using Phi-4's Mixture-of-LoRA speech adapter. |

