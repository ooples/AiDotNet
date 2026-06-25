---
title: "FireRedASR<T>"
description: "FireRedASR: fire-and-reduce dual-pass ASR system"
section: "API Reference"
---

`Models & Types` · `AiDotNet.SpeechRecognition.LLMIntegrated`

FireRedASR: fire-and-reduce dual-pass ASR system

## For Beginners

FireRedASR is an industrial-grade ASR system using a dual-pass architecture. The first pass (Fire) uses a fast Conformer-CTC encoder for streaming results. The second pass (Reduce) refines the output using attention-based rescoring with language m...

## How It Works

**References:**

- Paper: "FireRedASR: Open-Source Industrial-Grade Mandarin Speech Recognition" (FireRed Team, 2025)

FireRedASR is an industrial-grade ASR system using a dual-pass architecture. The first pass (Fire) uses a fast Conformer-CTC encoder for streaming results. The second pass (Reduce) refines the output using attention-based rescoring with language model integration. The dual-pass design balances latency and accuracy for production deployments. Achieves state-of-the-art on Mandarin ASR benchmarks with robust performance on accented and noisy speech.

## Methods

| Method | Summary |
|:-----|:--------|
| `CTCGreedyDecodeWithConfidence(Tensor<>)` | CTC greedy decode with per-frame softmax confidence tracking. |
| `ClassifyLanguageFromTokens(List<Int32>)` | Classifies language from decoded token distribution using script detection heuristics. |
| `ExtractSegments(String,Double,List<Int32>,Double)` | Extracts timestamped segments using proportional token-position alignment. |
| `TokensToText(List<Int32>)` | Converts token IDs to text with full Unicode support. |
| `Transcribe(Tensor<>,String,Boolean)` | Transcribes audio using FireRedASR's dual-pass architecture. |

