---
title: "TeacherModelType"
description: "Specifies the type of teacher model to use for knowledge distillation."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums`

Specifies the type of teacher model to use for knowledge distillation.

## For Beginners

The teacher model is the "expert" that guides the student model's learning.
Different teacher types are suited for different scenarios and distillation goals.

## How It Works

**Choosing a Teacher:**

- Use **NeuralNetwork** for standard NN-to-NN distillation
- Use **Ensemble** to combine knowledge from multiple models
- Use **Pretrained** to load from checkpoints or ONNX
- Use **Adaptive** for curriculum learning (progressive difficulty)
- Use **Online** when teacher should update during training

## Fields

| Field | Summary |
|:-----|:--------|
| `Adaptive` | Adaptive teacher that adjusts teaching based on student performance. |
| `Curriculum` | Curriculum teacher that provides progressive difficulty. |
| `Distributed` | Distributed teacher split across multiple devices/nodes. |
| `Ensemble` | Ensemble of multiple teacher models. |
| `MultiModal` | Multi-modal teacher (e.g., CLIP, vision-language models). |
| `NeuralNetwork` | Standard neural network teacher. |
| `Online` | Online teacher that updates during student training. |
| `Pretrained` | Pretrained model loaded from checkpoint or ONNX. |
| `Quantized` | Quantized teacher with reduced precision (INT8, INT4). |
| `Self` | Self-teacher where model teaches itself (Born-Again Networks). |
| `Transformer` | Transformer-based teacher (BERT, GPT, ViT, etc.). |

