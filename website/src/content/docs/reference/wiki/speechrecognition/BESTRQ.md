---
title: "BESTRQ<T>"
description: "BEST-RQ: BERT-based self-supervised learning with random-projection quantizer"
section: "API Reference"
---

`Models & Types` · `AiDotNet.SpeechRecognition.Foundation`

BEST-RQ: BERT-based self-supervised learning with random-projection quantizer

## For Beginners

BEST-RQ simplifies self-supervised speech pre-training by replacing learned codebooks with a fixed random-projection quantizer. Input speech features are projected by a random matrix and nearest-neighbor quantized to provide targets for masked pre...

## How It Works

**References:**

- Paper: "Self-supervised Learning with Random-Projection Quantizer for Speech Recognition" (Chiu et al., Google, 2022)

BEST-RQ simplifies self-supervised speech pre-training by replacing learned codebooks with a fixed random-projection quantizer. Input speech features are projected by a random matrix and nearest-neighbor quantized to provide targets for masked prediction. This eliminates the complex codebook learning of wav2vec 2.0 and the iterative clustering of HuBERT, while achieving competitive ASR performance. The simplicity enables efficient pre-training at scale.

## Methods

| Method | Summary |
|:-----|:--------|
| `Transcribe(Tensor<>,String,Boolean)` | Transcribes audio using BEST-RQ's random-projection quantizer encoder with CTC. |

