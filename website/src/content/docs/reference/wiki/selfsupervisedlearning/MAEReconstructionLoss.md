---
title: "MAEReconstructionLoss<T>"
description: "MAE (Masked Autoencoder) Reconstruction Loss for self-supervised learning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.SelfSupervisedLearning.Losses`

MAE (Masked Autoencoder) Reconstruction Loss for self-supervised learning.

## For Beginners

MAE loss measures how well the model reconstructs
masked patches of an image. Only the masked patches contribute to the loss,
making it efficient and focused on learning useful representations.

## How It Works

**Key insight:** By masking a large portion (75%) of patches and
reconstructing only those patches, the model learns rich visual representations
without requiring contrastive learning or negative samples.

**Loss formula:**

where M is the number of masked patches.

**Reference:** He et al., "Masked Autoencoders Are Scalable Vision Learners"
(CVPR 2022)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MAEReconstructionLoss(Boolean,Boolean)` | Initializes a new instance of the MAEReconstructionLoss class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AiDotNet#Interfaces#IContrastiveLoss{T}#ComputeLoss(Tensor<>,Tensor<>)` | IContrastiveLoss implementation — computes reconstruction loss with all-ones mask (all patches contribute equally). |
| `ComputeLoss(Tensor<>,Tensor<>,Tensor<>)` | Computes the reconstruction loss for masked patches. |
| `ComputeLossWithGradients(Tensor<>,Tensor<>,Tensor<>)` | Computes reconstruction loss with gradients for backpropagation. |
| `ComputePerSampleLoss(Tensor<>,Tensor<>,Tensor<>)` | Computes per-sample reconstruction loss (useful for analysis). |
| `CreateRandomMask(Int32,Int32,Double,Nullable<Int32>)` | Creates a random mask for patches. |

