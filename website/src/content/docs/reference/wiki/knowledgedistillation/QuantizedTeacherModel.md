---
title: "QuantizedTeacherModel<T>"
description: "Quantized teacher model with reduced precision for efficient deployment."
section: "API Reference"
---

`Models & Types` · `AiDotNet.KnowledgeDistillation.Teachers`

Quantized teacher model with reduced precision for efficient deployment.

## For Beginners

Quantization reduces the numerical precision of model weights
and activations to use fewer bits (e.g., 8-bit instead of 32-bit floating point).
This enables:

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `QuantizedTeacherModel(ITeacherModel<Vector<>,Vector<>>,Int32)` | Initializes a new instance of QuantizedTeacherModel wrapping a teacher interface. |

## Properties

| Property | Summary |
|:-----|:--------|
| `OutputDimension` | Gets the output dimension of the teacher model. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetLogits(Vector<>)` | Gets quantized logits from the teacher model. |
| `QuantizeDynamic(Vector<>)` | Applies dynamic quantization (per-batch min/max). |
| `QuantizeFixedScale(Vector<>)` | Applies fixed-scale quantization (JIT-compatible). |

