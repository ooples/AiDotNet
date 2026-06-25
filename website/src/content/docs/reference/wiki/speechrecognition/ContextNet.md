---
title: "ContextNet<T>"
description: "ContextNet: CNN encoder with squeeze-and-excitation and global context for ASR."
section: "API Reference"
---

`Models & Types` · `AiDotNet.SpeechRecognition.ConformerFamily`

ContextNet: CNN encoder with squeeze-and-excitation and global context for ASR.

## For Beginners

A purely convolutional encoder that uses squeeze-and-excitation (SE) blocks to capture global context. Each block contains depthwise separable convolutions with SE modules that adaptively reweight channel features based on the entire sequence. Typ...

## How It Works

**References:**

- Paper: "ContextNet: Improving Convolutional Neural Networks for ASR with Global Context" (Han et al., 2020)

A purely convolutional encoder that uses squeeze-and-excitation (SE) blocks to capture
global context. Each block contains depthwise separable convolutions with SE modules
that adaptively reweight channel features based on the entire sequence. Typically paired
with an RNN-T decoder for streaming ASR. Achieves WER 1.9%/3.9% on LibriSpeech.

## Methods

| Method | Summary |
|:-----|:--------|
| `TokensToText(List<Int32>)` | Maps token IDs to text. |
| `Transcribe(Tensor<>,String,Boolean)` | Transcribes audio using ContextNet's CNN encoder with squeeze-and-excitation. |

