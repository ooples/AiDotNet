---
title: "EmformerRNNT<T>"
description: "Emformer-RNNT: efficient memory Transformer for streaming ASR"
section: "API Reference"
---

`Models & Types` · `AiDotNet.SpeechRecognition.Streaming`

Emformer-RNNT: efficient memory Transformer for streaming ASR

## For Beginners

Emformer introduces an efficient memory mechanism for streaming Transformer ASR. The model processes audio segments and maintains a fixed-size memory bank that summarizes past context. Each segment attends to its own frames plus the memory bank, a...

## How It Works

**References:**

- Paper: "Emformer: Efficient Memory Transformer Based Acoustic Model for Low Latency Streaming Speech Recognition" (Shi et al., Meta, 2021)

Emformer introduces an efficient memory mechanism for streaming Transformer ASR. The model processes audio segments and maintains a fixed-size memory bank that summarizes past context. Each segment attends to its own frames plus the memory bank, avoiding the quadratic cost of attending to the full history. The memory is updated via a learned summarization operation after each segment. Combined with an RNN-T decoder, Emformer achieves low-latency streaming with competitive accuracy.

## Methods

| Method | Summary |
|:-----|:--------|
| `Transcribe(Tensor<>,String,Boolean)` | Transcribes audio using Emformer's memory-augmented streaming architecture. |

