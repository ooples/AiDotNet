---
title: "ParakeetTDT<T>"
description: "Parakeet-TDT: NVIDIA NeMo's 1.1B Conformer with Token-and-Duration Transducer."
section: "API Reference"
---

`Models & Types` · `AiDotNet.SpeechRecognition.NeMo`

Parakeet-TDT: NVIDIA NeMo's 1.1B Conformer with Token-and-Duration Transducer.

## For Beginners

Parakeet-TDT extends the standard RNN-T by jointly predicting both the output token and the number of encoder frames to advance (duration). This "Token-and-Duration" formulation allows the model to skip multiple blank frames at once, significantly...

## How It Works

**References:**

- Paper: "Token-and-Duration Transducer for streaming ASR" (NVIDIA, 2024)

Parakeet-TDT extends the standard RNN-T by jointly predicting both the output token and
the number of encoder frames to advance (duration). This "Token-and-Duration" formulation
allows the model to skip multiple blank frames at once, significantly reducing inference
latency for streaming. The joint network now outputs both token logits and duration logits.
During greedy decode, when a non-blank token is emitted, the duration head specifies how
many frames to skip forward, avoiding per-frame blank predictions.

## Methods

| Method | Summary |
|:-----|:--------|
| `TDTGreedyDecodeWithConfidence(Tensor<>)` | TDT greedy decode with confidence: emits token + skips frames based on duration prediction. |
| `Transcribe(Tensor<>,String,Boolean)` | Transcribes audio using Fast Conformer encoder + TDT decoder. |

