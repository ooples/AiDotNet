---
title: "QATMethod"
description: "Specifies the Quantization-Aware Training (QAT) method to use during model training."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums`

Specifies the Quantization-Aware Training (QAT) method to use during model training.
QAT simulates quantization effects during training so the model learns to be robust to low precision.

## For Beginners

Normally, we train a model in full precision (32-bit) and then
compress it afterward (PTQ). QAT is smarter - it simulates the compression DURING training,
so the model learns to work well even when compressed.

## How It Works

**Analogy:** It's like training for a marathon at high altitude - when you compete
at sea level, you perform better because you trained under harder conditions.

**QAT vs PTQ Comparison:**

**Research References:**

## Fields

| Field | Summary |
|:-----|:--------|
| `EfficientQAT` | EfficientQAT - memory-efficient QAT optimized for large language models. |
| `ParetoQ` | ParetoQ - state-of-the-art QAT achieving optimal accuracy across all bit widths. |
| `QABLoRA` | QA-BLoRA - Quantization-Aware fine-tuning with Balanced Low-Rank Adaptation. |
| `Standard` | Standard QAT using Straight-Through Estimator (STE) for gradient propagation. |
| `ZeroQAT` | ZeroQAT - zeroth-order optimization based QAT that doesn't require backpropagation. |

