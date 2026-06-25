---
title: "SSLMethodType"
description: "Specifies the type of self-supervised learning method to use for representation learning."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums`

Specifies the type of self-supervised learning method to use for representation learning.

## For Beginners

Self-supervised learning (SSL) methods learn useful representations
from unlabeled data by creating "pretext tasks" - artificial tasks that force the model to learn
meaningful features. Different methods use different strategies to achieve this.

## How It Works

**Choosing a Method:**

- Use **SimCLR** for simplicity and good performance (no memory bank needed)
- Use **MoCo** variants for large batch sizes or limited GPU memory
- Use **BYOL** or **SimSiam** to avoid negative sample mining
- Use **BarlowTwins** for interpretable redundancy-reduction approach
- Use **DINO** for Vision Transformers with self-distillation
- Use **MAE** for generative masked autoencoding approach

## Fields

| Field | Summary |
|:-----|:--------|
| `BYOL` | BYOL: Bootstrap Your Own Latent (Grill et al., 2020). |
| `BarlowTwins` | Barlow Twins: Self-Supervised Learning via Redundancy Reduction (Zbontar et al., 2021). |
| `DINO` | DINO: Emerging Properties in Self-Supervised Vision Transformers (Caron et al., 2021). |
| `MAE` | MAE: Masked Autoencoders Are Scalable Vision Learners (He et al., 2022). |
| `MoCo` | MoCo: Momentum Contrast for Unsupervised Visual Representation Learning (He et al., 2020). |
| `MoCoV2` | MoCo v2: Improved Baselines with Momentum Contrastive Learning (Chen et al., 2020). |
| `MoCoV3` | MoCo v3: An Empirical Study of Training Self-Supervised Vision Transformers (Chen et al., 2021). |
| `SimCLR` | SimCLR: A Simple Framework for Contrastive Learning of Visual Representations (Chen et al., 2020). |
| `iBOT` | iBOT: Image BERT Pre-Training with Online Tokenizer (Zhou et al., 2022). |

