---
title: "CanaryFlash<T>"
description: "Canary-Flash: NVIDIA NeMo lightweight multilingual ASR + translation with hybrid CTC/attention."
section: "API Reference"
---

`Models & Types` · `AiDotNet.SpeechRecognition.NeMo`

Canary-Flash: NVIDIA NeMo lightweight multilingual ASR + translation with hybrid CTC/attention.

## For Beginners

Canary-Flash is a smaller, faster variant of the Canary multilingual ASR family. It uses a Fast Conformer encoder with a lightweight attention-based encoder-decoder (AED), not a full LLM. The hybrid CTC/attention training uses both a CTC auxiliary...

## How It Works

**References:**

- Model: "Canary-Flash" (NVIDIA NeMo, 2025)

Canary-Flash is a smaller, faster variant of the Canary multilingual ASR family. It uses
a Fast Conformer encoder with a lightweight attention-based encoder-decoder (AED), not a
full LLM. The hybrid CTC/attention training uses both a CTC auxiliary loss on the encoder
output and cross-entropy loss on the decoder, improving robustness. Flash attention and
quantization-aware training enable faster inference. Supports 20+ languages for ASR and
speech translation with lower latency than Canary-Qwen.

## Methods

| Method | Summary |
|:-----|:--------|
| `Transcribe(Tensor<>,String,Boolean)` | Transcribes audio using hybrid CTC/attention encoder-decoder. |

