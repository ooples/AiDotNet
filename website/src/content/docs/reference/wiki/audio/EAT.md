---
title: "EAT<T>"
description: "EAT (Efficient Audio Transformer) model for efficient audio event detection and classification."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Audio.Classification`

EAT (Efficient Audio Transformer) model for efficient audio event detection and classification.

## For Beginners

EAT is like having a wise teacher and a student. The teacher sees
the full spectrogram and the student only sees 25% of it. The student must predict what
the teacher sees. As the student improves, the teacher slowly updates to match, creating
a virtuous learning cycle. This is more efficient than BEATs because the student processes
fewer patches.

**Usage:**

## How It Works

EAT (Chen et al., 2024) achieves competitive performance with significantly less compute
than BEATs through an efficient self-supervised pre-training approach using teacher-student
distillation. It reaches 49.7% mAP on AudioSet-2M using only 10% of BEATs' compute.

**Architecture:** EAT uses a standard ViT-Base encoder with an EMA teacher:

- **Student encoder**: 12-layer Transformer that processes visible (unmasked) patches
- **Teacher encoder**: EMA copy of student that processes all patches, providing targets
- **Masked prediction**: Student predicts teacher's representations for masked patches

**References:**

- Paper: "EAT: Self-Supervised Pre-Training with Efficient Audio Transformer" (Chen et al., 2024)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `EAT(NeuralNetworkArchitecture<>,EATOptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates an EAT model for native training mode. |
| `EAT(NeuralNetworkArchitecture<>,String,EATOptions)` | Creates an EAT model for ONNX inference mode. |

