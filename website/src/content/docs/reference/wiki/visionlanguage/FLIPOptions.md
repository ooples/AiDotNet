---
title: "FLIPOptions"
description: "Configuration options for the FLIP (Fast Language-Image Pre-training) model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.VisionLanguage.Encoders`

Configuration options for the FLIP (Fast Language-Image Pre-training) model.

## For Beginners

FLIP makes CLIP training much faster by a simple trick: during training,
it randomly hides most of the image (like covering parts of a puzzle) and only processes the
visible pieces. This speeds up training 2-4x with minimal performance loss, because the model
learns to understand images from partial information.

## How It Works

FLIP (Li et al., 2022) accelerates CLIP training by randomly masking 50-75% of image patches
during training. The unmasked patches are processed by the vision encoder, reducing computation
while maintaining strong zero-shot performance. At inference time, all patches are used.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FLIPOptions` | Initializes default FLIP options. |
| `FLIPOptions(FLIPOptions)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `LossType` | Gets or sets the contrastive loss type. |
| `MaskingRatio` | Gets or sets the masking ratio for image patches during training. |
| `UnmaskedTuningEpochs` | Gets or sets the number of unmasked tuning epochs. |
| `UseUnmaskedTuning` | Gets or sets whether to use unmasked tuning after pre-training. |

