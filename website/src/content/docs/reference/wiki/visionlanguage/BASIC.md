---
title: "BASIC<T>"
description: "BASIC (Combined Scaling for Zero-shot Transfer Learning) model using CoAtNet vision encoder."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.Encoders`

BASIC (Combined Scaling for Zero-shot Transfer Learning) model using CoAtNet vision encoder.

## For Beginners

BASIC pushes CLIP performance higher by combining three things at
scale: a stronger vision architecture (CoAtNet, which mixes convolutions and attention),
a larger dataset, and bigger batch sizes during training. The result is 85.7% zero-shot
ImageNet accuracy. Default values follow the original paper settings.

## How It Works

BASIC (Pham et al., 2022) combines CoAtNet (Convolution + Attention) vision encoder with
large-scale contrastive image-text pre-training, achieving 85.7% zero-shot ImageNet via
architecture+data+batch-size combined scaling.

**References:**

- Paper: "Combined Scaling for Zero-shot Transfer Learning" (Pham et al., 2022)

## Methods

| Method | Summary |
|:-----|:--------|
| `GetExtraTrainableLayers` |  |

