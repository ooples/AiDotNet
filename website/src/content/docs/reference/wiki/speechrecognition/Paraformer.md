---
title: "Paraformer<T>"
description: "Paraformer: fast and accurate parallel Transformer for non-autoregressive ASR"
section: "API Reference"
---

`Models & Types` · `AiDotNet.SpeechRecognition.AlibabaASR`

Paraformer: fast and accurate parallel Transformer for non-autoregressive ASR

## For Beginners

Paraformer uses Continuous Integrate-and-Fire (CIF) to predict token counts and extract acoustic embeddings in a single forward pass, enabling non-autoregressive parallel decoding. The CIF module accumulates encoder hidden states weighted by learn...

## How It Works

**References:**

- Paper: "Paraformer: Fast and Accurate Parallel Transformer for Non-autoregressive End-to-End Speech Recognition" (Gao et al., Alibaba DAMO, 2022)

Paraformer uses Continuous Integrate-and-Fire (CIF) to predict token counts and extract acoustic embeddings in a single forward pass, enabling non-autoregressive parallel decoding. The CIF module accumulates encoder hidden states weighted by learned firing probabilities. When cumulative weight exceeds a threshold, an acoustic embedding is emitted. A glancing language model (GLM) decoder then generates all tokens in parallel from these embeddings. This achieves comparable accuracy to autoregressive models with much lower latency.

## Methods

| Method | Summary |
|:-----|:--------|
| `Transcribe(Tensor<>,String,Boolean)` | Transcribes audio using CIF-based parallel decoding. |

