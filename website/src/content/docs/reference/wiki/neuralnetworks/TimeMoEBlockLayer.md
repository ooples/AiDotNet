---
title: "TimeMoEBlockLayer<T>"
description: "A single Time-MoE transformer block: multi-head self-attention + Mixture-of-Experts FFN, each wrapped in pre-norm + residual."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers`

A single Time-MoE transformer block: multi-head self-attention + Mixture-of-Experts FFN,
each wrapped in pre-norm + residual. Per Shi et al. 2024 "Time-MoE: Billion-Scale Time Series
Foundation Models with Mixture of Experts".

## How It Works

Forward sequence (pre-norm, GPT-style):

- norm1(input) → self-attention → residual add with input → x
- norm2(x) → MoE-FFN (top-k expert dispatch) → residual add with x → output

The MoE routes each token (each row of the flattened [B·numPatches, hiddenDim] tensor)
independently through top-k experts, per paper. Each expert is a 2-layer Dense FFN with
GELU (hiddenDim → intermediateSize → hiddenDim). Load-balancing is enabled with weight
0.01.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TimeMoEBlockLayer(Int32,Int32,Int32,Int32,Int32)` | Initializes a new `TimeMoEBlockLayer`. |

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportsTraining` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Forward(Tensor<>)` |  |
| `GetAuxiliaryLoss` | Surfaces the MoE router's load-balancing auxiliary loss for the current (most recent) forward pass. |
| `GetMetadata` | Persists the constructor arguments so the deserializer can rebuild this layer at the same shape. |
| `GetParameters` |  |
| `ResetState` |  |
| `SetParameters(Vector<>)` |  |
| `UpdateParameters()` |  |

