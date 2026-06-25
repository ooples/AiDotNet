---
title: "ActivationCategory"
description: "Categories of activation functions based on their architectural role and behavior."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums`

Categories of activation functions based on their architectural role and behavior.

## Fields

| Field | Summary |
|:-----|:--------|
| `Gate` | Gate activations that control information flow (Sigmoid for LSTM gates, Tanh for cell state). |
| `General` | General-purpose activations suitable for hidden layers (ReLU, GELU, Swish). |
| `Normalization` | Normalization activations that produce probability distributions (Softmax, Sparsemax, LogSoftmax). |
| `Output` | Output activations that produce final predictions (Softmax for classification, Sigmoid for binary). |
| `Parametric` | Parametric activations with learnable parameters (PReLU, Maxout). |
| `Stochastic` | Stochastic activations with random components (GumbelSoftmax, BinarySpiking, RReLU). |

