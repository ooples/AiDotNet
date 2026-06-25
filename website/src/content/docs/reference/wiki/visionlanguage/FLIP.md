---
title: "FLIP<T>"
description: "FLIP (Fast Language-Image Pre-training) model that accelerates CLIP training via random masking of image patches."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.Encoders`

FLIP (Fast Language-Image Pre-training) model that accelerates CLIP training via
random masking of image patches.

## For Beginners

FLIP speeds up CLIP training by randomly hiding 50-75% of
image patches during training, dramatically reducing computation. At inference time, all
patches are used for full accuracy. This simple trick allows training larger models or
using more data within the same compute budget. Default values follow the original paper
settings.

## How It Works

FLIP (Li et al., 2022) randomly masks 50-75% of image patches during training, reducing
computation while maintaining strong zero-shot performance. At inference, all patches are used.

**References:**

- Paper: "Scaling Language-Image Pre-training via Masking" (Li et al., 2022)

## Methods

| Method | Summary |
|:-----|:--------|
| `GetExtraTrainableLayers` |  |

