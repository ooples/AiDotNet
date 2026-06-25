---
title: "ContrastiveLossType"
description: "Specifies the contrastive loss function type."
section: "API Reference"
---

`Enums` · `AiDotNet.VisionLanguage.Encoders`

Specifies the contrastive loss function type.

## Fields

| Field | Summary |
|:-----|:--------|
| `InfoNCE` | Standard InfoNCE (softmax) loss used by CLIP. |
| `Sigmoid` | Sigmoid loss used by SigLIP for pairwise computation (no global normalization). |
| `SymmetricCrossEntropy` | Symmetric cross-entropy used by OpenCLIP. |

