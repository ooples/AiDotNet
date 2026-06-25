---
title: "Branchformer<T>"
description: "Branchformer speech recognition model with parallel MLP-attention branches."
section: "API Reference"
---

`Models & Types` · `AiDotNet.SpeechRecognition.ConformerFamily`

Branchformer speech recognition model with parallel MLP-attention branches.

## For Beginners

The Branchformer processes audio through two parallel branches per layer: (1) Multi-head self-attention for global context, (2) Convolutional gating MLP (cgMLP) for local patterns. The branches are concatenated and merged with a learned linear pro...

## How It Works

**References:**

- Paper: "Branchformer: Parallel MLP-Attention Architectures to Achieve High Accuracy and Linear-Time Complexity for Speech Processing" (Peng et al., 2022)

The Branchformer processes audio through two parallel branches per layer:
(1) Multi-head self-attention for global context,
(2) Convolutional gating MLP (cgMLP) for local patterns.
The branches are concatenated and merged with a learned linear projection,
allowing each layer to capture both local and global dependencies simultaneously.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `Branchformer(NeuralNetworkArchitecture<>,BranchformerOptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates a Branchformer model in native training mode. |
| `Branchformer(NeuralNetworkArchitecture<>,String,BranchformerOptions)` | Creates a Branchformer model in ONNX inference mode. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Transcribe(Tensor<>,String,Boolean)` | Transcribes audio using Branchformer's parallel-branch encoder with CTC decoding. |

