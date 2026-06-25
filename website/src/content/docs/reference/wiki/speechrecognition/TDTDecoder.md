---
title: "TDTDecoder<T>"
description: "TDT Decoder: Token-and-Duration Transducer for efficient streaming"
section: "API Reference"
---

`Models & Types` · `AiDotNet.SpeechRecognition.Streaming`

TDT Decoder: Token-and-Duration Transducer for efficient streaming

## For Beginners

The TDT (Token-and-Duration Transducer) decoder extends standard RNN-T by jointly predicting both the output token and the number of encoder frames to skip. When a non-blank token is emitted, the duration head predicts how many blank frames to ski...

## How It Works

**References:**

- Paper: "Efficient Sequence Transduction by Jointly Predicting Tokens and Durations" (Xu et al., NVIDIA, 2023)

The TDT (Token-and-Duration Transducer) decoder extends standard RNN-T by jointly predicting both the output token and the number of encoder frames to skip. When a non-blank token is emitted, the duration head predicts how many blank frames to skip, reducing the number of joint network forward passes. This achieves up to 2.5x inference speedup over standard RNN-T without accuracy degradation. The approach is orthogonal to encoder optimization and combines well with Fast Conformer.

## Methods

| Method | Summary |
|:-----|:--------|
| `Transcribe(Tensor<>,String,Boolean)` | Transcribes audio using TDT's joint token-duration prediction. |

