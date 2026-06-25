---
title: "LossCategory"
description: "Categories of loss functions based on the type of learning task they serve."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums`

Categories of loss functions based on the type of learning task they serve.

## Fields

| Field | Summary |
|:-----|:--------|
| `Adversarial` | Adversarial losses (Wasserstein, Hinge for GANs). |
| `Classification` | Classification losses (CrossEntropy, Focal, Hinge). |
| `Contrastive` | Contrastive/self-supervised losses (InfoNCE, SimCLR, BYOL). |
| `Generation` | Generative model losses (Adversarial, Perceptual, Reconstruction). |
| `PhysicsInformed` | Physics-informed losses (PDE residual, boundary condition). |
| `Ranking` | Ranking/metric learning losses (Triplet, Contrastive, MarginRanking). |
| `Reconstruction` | Reconstruction losses (MSE, BCE for autoencoders/VAEs). |
| `Regression` | Regression losses (MSE, MAE, Huber). |
| `Regularization` | Regularization losses (L1, L2, KL divergence). |
| `Segmentation` | Segmentation losses (Dice, Tversky, IoU). |

