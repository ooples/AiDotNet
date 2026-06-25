---
title: "AudioPaLM<T>"
description: "AudioPaLM: large language model for speech understanding and generation"
section: "API Reference"
---

`Models & Types` · `AiDotNet.SpeechRecognition.LLMIntegrated`

AudioPaLM: large language model for speech understanding and generation

## For Beginners

AudioPaLM fuses the PaLM-2 text language model with AudioLM's speech processing capabilities into a single multimodal model. For ASR, speech is encoded using USM encoder features, which are projected into PaLM-2's embedding space. The model genera...

## How It Works

**References:**

- Paper: "AudioPaLM: A Large Language Model That Can Speak and Listen" (Rubenstein et al., Google, 2023)

AudioPaLM fuses the PaLM-2 text language model with AudioLM's speech processing capabilities into a single multimodal model. For ASR, speech is encoded using USM encoder features, which are projected into PaLM-2's embedding space. The model generates text tokens conditioned on the speech input, leveraging PaLM-2's massive language understanding. AudioPaLM achieves state-of-the-art on speech translation while maintaining competitive ASR performance.

## Methods

| Method | Summary |
|:-----|:--------|
| `Transcribe(Tensor<>,String,Boolean)` | Transcribes audio using AudioPaLM's fused speech-text architecture. |

