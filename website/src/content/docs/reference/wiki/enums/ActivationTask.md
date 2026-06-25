---
title: "ActivationTask"
description: "Tasks or architectural positions where an activation function is commonly used."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums`

Tasks or architectural positions where an activation function is commonly used.

## Fields

| Field | Summary |
|:-----|:--------|
| `AttentionGating` | Attention mechanism gating (Softmax over attention scores). |
| `CapsuleSquash` | Capsule network squashing functions. |
| `GenerativeOutput` | Generative model outputs (Tanh for image generation [-1,1]). |
| `HiddenLayer` | Standard hidden layer activation in feedforward/convolutional networks. |
| `NormalizationOutput` | Producing normalized probability distributions. |
| `OutputLayer` | Output layer activation for final predictions (classification, regression). |
| `RecurrentGating` | Recurrent network gating (LSTM/GRU forget/input/output gates). |
| `SpikingNeuron` | Spiking neural network activations. |
| `TransformerFFN` | Transformer feed-forward sublayers (GELU, SwiGLU). |

