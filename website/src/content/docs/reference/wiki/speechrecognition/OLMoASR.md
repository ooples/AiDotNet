---
title: "OLMoASR<T>"
description: "OLMo-ASR: open language model adapted for speech recognition"
section: "API Reference"
---

`Models & Types` · `AiDotNet.SpeechRecognition.LLMIntegrated`

OLMo-ASR: open language model adapted for speech recognition

## For Beginners

OLMo-ASR adapts the open-source OLMo language model for speech recognition by adding a speech encoder adapter. A Conformer encoder processes audio, then a learned adapter projects encoder representations into the LLM's embedding space. The LLM dec...

## How It Works

**References:**

- Paper: "OLMo: Accelerating the Science of Language Models" (Groeneveld et al., 2024)

OLMo-ASR adapts the open-source OLMo language model for speech recognition by adding a speech encoder adapter. A Conformer encoder processes audio, then a learned adapter projects encoder representations into the LLM's embedding space. The LLM decoder generates text autoregressively conditioned on the speech embeddings, leveraging its strong language understanding for accurate transcription.

## Methods

| Method | Summary |
|:-----|:--------|
| `Transcribe(Tensor<>,String,Boolean)` | Transcribes audio using Conformer encoder + OLMo LLM decoder. |

