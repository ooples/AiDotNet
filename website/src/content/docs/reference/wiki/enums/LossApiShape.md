---
title: "LossApiShape"
description: "Describes the method signature shape a loss function uses for its primary calculation."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums`

Describes the method signature shape a loss function uses for its primary calculation.
The test scaffold generator uses this to select the correct test base class.

## Fields

| Field | Summary |
|:-----|:--------|
| `ComplexInterleaved` | Standard CalculateLoss(Vector, Vector) but inputs are complex-interleaved pairs [real, imag, real, imag, ...]. |
| `ImageMatrix` | Image-based Calculate(Matrix, Matrix) requiring a feature extractor. |
| `PairedEmbedding` | Paired embedding CalculateLoss(Vector, Vector, T) with two embedding vectors and a similarity label. |
| `SelfSupervised` | Self-supervised CreateTask interface, not a standard loss calculation. |
| `SparseIndex` | Standard CalculateLoss(Vector, Vector) but predicted and actual have different lengths. |
| `TargetNoiseMatrix` | Contrastive-style Calculate(Vector, Matrix) with target logits and noise logits. |
| `TripletMatrix` | Triplet-style CalculateLoss(Matrix, Matrix, Matrix) with anchor, positive, negative. |
| `VectorVector` | Standard CalculateLoss(Vector, Vector) interface. |

