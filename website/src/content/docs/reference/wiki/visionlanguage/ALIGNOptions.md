---
title: "ALIGNOptions"
description: "Configuration options for the ALIGN (A Large-scale ImaGe and Noisy-text embedding) model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.VisionLanguage.Encoders`

Configuration options for the ALIGN (A Large-scale ImaGe and Noisy-text embedding) model.

## For Beginners

ALIGN is similar to CLIP but uses a different image processing backbone
(EfficientNet instead of ViT) and was trained on a much larger but noisier dataset. The key
insight is that scale can compensate for noise in training data.

## How It Works

ALIGN (Jia et al., ICML 2021) demonstrates that a simple dual-encoder contrastive model
can achieve strong vision-language alignment when trained on a massive noisy dataset of
1.8 billion image-alt-text pairs from the web. Unlike CLIP which uses a ViT,
ALIGN uses an EfficientNet as its vision encoder.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ALIGNOptions` | Initializes default ALIGN options. |
| `ALIGNOptions(ALIGNOptions)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `EfficientNetCompoundCoefficient` | Gets or sets the EfficientNet compound scaling coefficient. |
| `LossType` | Gets or sets the contrastive loss type. |
| `UseSqueezeExcitation` | Gets or sets whether to use squeeze-and-excitation attention blocks in EfficientNet. |

