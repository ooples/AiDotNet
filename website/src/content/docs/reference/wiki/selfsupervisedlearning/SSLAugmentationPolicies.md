---
title: "SSLAugmentationPolicies<T>"
description: "Provides standard augmentation policies for self-supervised learning methods."
section: "API Reference"
---

`Models & Types` · `AiDotNet.SelfSupervisedLearning`

Provides standard augmentation policies for self-supervised learning methods.

## For Beginners

Self-supervised learning methods rely heavily on data augmentation
to create different "views" of the same image. These views should be different enough to be
challenging, but similar enough that they can still be recognized as the same image.

## How It Works

**Common augmentations for SSL:**

**Method-specific policies:**

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SSLAugmentationPolicies(Nullable<Int32>)` | Initializes a new instance of the SSLAugmentationPolicies class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyBYOL(Tensor<>)` | Applies BYOL-style asymmetric augmentation. |
| `ApplyDINO(Tensor<>,Int32,Int32)` | Applies DINO-style multi-crop augmentation. |
| `ApplyMinimal(Tensor<>)` | Applies minimal augmentation (used by MAE which relies on masking rather than augmentation). |
| `ApplyMoCoV2(Tensor<>)` | Applies MoCo v2 style augmentation. |
| `ApplySimCLR(Tensor<>)` | Applies SimCLR-style augmentation to create two views. |
| `ForBYOL(Nullable<Int32>)` | Creates a standard BYOL augmentation policy. |
| `ForDINO(Nullable<Int32>)` | Creates a standard DINO augmentation policy. |
| `ForSimCLR(Nullable<Int32>)` | Creates a standard SimCLR augmentation policy. |

