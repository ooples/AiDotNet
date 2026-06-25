---
title: "LiT<T>"
description: "LiT (Locked-image Tuning) model that freezes a pre-trained image encoder and only trains the text encoder for efficient contrastive image-text alignment."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.Encoders`

LiT (Locked-image Tuning) model that freezes a pre-trained image encoder and only
trains the text encoder for efficient contrastive image-text alignment.

## For Beginners

LiT makes CLIP training much cheaper by freezing a pre-trained
image encoder and only training the text encoder to align with it. Instead of training
both vision and language components from scratch, it reuses an existing strong vision model
and teaches a text model to "read" its image features, achieving 85.2% zero-shot ImageNet
accuracy with much less compute. Default values follow the original paper settings.

## How It Works

LiT (Zhai et al., CVPR 2022) demonstrates that contrastive image-text training can be made
dramatically more efficient by freezing a high-quality pre-trained vision encoder (e.g., from
ImageNet-21k) and only training the text encoder to align with it. This "locked-image tuning"
achieves 85.2% zero-shot accuracy on ImageNet with significantly reduced training cost.

**References:**

- Paper: "LiT: Zero-Shot Transfer with Locked-image text Tuning" (Zhai et al., CVPR 2022)

## Methods

| Method | Summary |
|:-----|:--------|
| `GetExtraTrainableLayers` |  |

