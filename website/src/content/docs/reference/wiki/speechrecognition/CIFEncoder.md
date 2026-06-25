---
title: "CIFEncoder<T>"
description: "CIF (Continuous Integrate-and-Fire) encoder for soft monotonic alignment in ASR."
section: "API Reference"
---

`Models & Types` · `AiDotNet.SpeechRecognition.ConformerFamily`

CIF (Continuous Integrate-and-Fire) encoder for soft monotonic alignment in ASR.

## For Beginners

CIF provides a soft, monotonic alignment mechanism between encoder frames and output tokens. Each encoder frame produces a firing weight; when cumulative weight exceeds a threshold (1.0), the accumulated weighted features "fire" as an output token...

## How It Works

**References:**

- Paper: "CIF: Continuous Integrate-and-Fire for End-to-End Speech Recognition" (Dong & Xu, 2020)

CIF provides a soft, monotonic alignment mechanism between encoder frames and output tokens.
Each encoder frame produces a firing weight; when cumulative weight exceeds a threshold (1.0),
the accumulated weighted features "fire" as an output token representation.
This enables non-autoregressive decoding (like Paraformer) while maintaining monotonic alignment.

## Methods

| Method | Summary |
|:-----|:--------|
| `Transcribe(Tensor<>,String,Boolean)` | Transcribes audio using CIF-based soft monotonic alignment. |

