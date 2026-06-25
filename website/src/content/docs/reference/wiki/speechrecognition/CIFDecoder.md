---
title: "CIFDecoder<T>"
description: "CIF: Continuous Integrate-and-Fire mechanism for non-autoregressive ASR"
section: "API Reference"
---

`Models & Types` · `AiDotNet.SpeechRecognition.CTCVariants`

CIF: Continuous Integrate-and-Fire mechanism for non-autoregressive ASR

## For Beginners

CIF (Continuous Integrate-and-Fire) is a mechanism that bridges the length mismatch between encoder frames and output tokens. The CIF module accumulates encoder hidden states weighted by learned firing probabilities. When the cumulative weight rea...

## How It Works

**References:**

- Paper: "CIF: Continuous Integrate-and-Fire for End-to-End Speech Recognition" (Dong and Xu, 2020)

CIF (Continuous Integrate-and-Fire) is a mechanism that bridges the length mismatch between encoder frames and output tokens. The CIF module accumulates encoder hidden states weighted by learned firing probabilities. When the cumulative weight reaches a threshold, an acoustic embedding is emitted and the accumulator resets. This produces exactly the right number of acoustic embeddings for the target sequence, enabling non-autoregressive parallel decoding without the conditional independence assumption of CTC.

## Methods

| Method | Summary |
|:-----|:--------|
| `Transcribe(Tensor<>,String,Boolean)` | Transcribes audio using CIF-based alignment with parallel decoding. |

