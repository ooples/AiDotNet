---
title: "ConstantScaleLayer<T>"
description: "Multiplies its input by a fixed (non-trainable) scalar."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers`

Multiplies its input by a fixed (non-trainable) scalar. Useful for
paper-canonical embedding rescaling — Vaswani 2017 §3.4 (preserved by
T5 / LLaMA / Gemma / Qwen2 / ChatGLM3) multiplies token embeddings by
√d_model before feeding them into the transformer stack.

## Properties

| Property | Summary |
|:-----|:--------|
| `ParameterCount` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ClearGradients` |  |
| `Forward(Tensor<>)` |  |
| `GetParameterGradients` |  |
| `GetParameters` |  |
| `ResetState` |  |
| `SetParameters(Vector<>)` |  |
| `UpdateParameters()` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `_scaleTensor` | Singleton scale held as a Tensor so Forward can use the same tape-tracked `Tensor{` path that RMSNorm uses for its γ gain. |

