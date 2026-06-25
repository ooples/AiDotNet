---
title: "SigLIPOptions"
description: "Configuration options for the SigLIP (Sigmoid Loss for Language-Image Pre-training) model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.VisionLanguage.Encoders`

Configuration options for the SigLIP (Sigmoid Loss for Language-Image Pre-training) model.

## For Beginners

SigLIP is like CLIP but with a smarter training strategy. Instead of
comparing all images with all texts at once (which gets expensive), it compares each image-text
pair independently. This makes it easier to train on very large batches and often gives
better results.

## How It Works

SigLIP (Zhai et al., ICCV 2023) replaces the standard softmax-based InfoNCE contrastive loss
with a sigmoid loss that operates on individual image-text pairs. This eliminates the need for
a global normalization across the batch, enabling better scaling to large batch sizes and
achieving 84.5% zero-shot accuracy on ImageNet with ViT-L/16@384.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SigLIPOptions` | Initializes default SigLIP options. |
| `SigLIPOptions(SigLIPOptions)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `CaptioningLossWeight` | Gets or sets the weight for the captioning loss (SigLIP 2). |
| `LossType` | Gets or sets the contrastive loss type (default: Sigmoid for SigLIP). |
| `Multilingual` | Gets or sets whether to enable multilingual text support (SigLIP 2 feature). |
| `SelfSupervisedLossWeight` | Gets or sets the weight for the self-supervised loss (SigLIP 2). |
| `SigmoidBias` | Gets or sets the bias term for sigmoid loss. |
| `UseSigLIP2` | Gets or sets whether to use the SigLIP 2 variant with additional loss terms. |

