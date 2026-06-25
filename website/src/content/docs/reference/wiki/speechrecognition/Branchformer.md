---
title: "Branchformer<T>"
description: "Branchformer: parallel attention and convolution branches"
section: "API Reference"
---

`Models & Types` · `AiDotNet.SpeechRecognition.CTCVariants`

Branchformer: parallel attention and convolution branches

## For Beginners

Branchformer processes each encoder block through two parallel branches: a self-attention branch for global context and a convolutional branch (cgMLP) for local patterns. The branches operate independently and their outputs are merged via a learne...

## How It Works

**References:**

- Paper: "Branchformer: Parallel MLP-Attention Architectures to Capture Local and Global Context for Speech Classification and Recognition" (Peng et al., 2022)

Branchformer processes each encoder block through two parallel branches: a self-attention branch for global context and a convolutional branch (cgMLP) for local patterns. The branches operate independently and their outputs are merged via a learned gating mechanism. This parallel design allows the model to capture both local acoustic details and long-range dependencies more effectively than Conformer's sequential attention-convolution design. Branchformer consistently outperforms Conformer on ASR benchmarks.

## Methods

| Method | Summary |
|:-----|:--------|
| `Transcribe(Tensor<>,String,Boolean)` | Transcribes audio using Branchformer's parallel attention-convolution encoder. |

