---
title: "UNITEROptions"
description: "Configuration options for UNITER (Universal Image-TExt Representation)."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.VisionLanguage.Foundational`

Configuration options for UNITER (Universal Image-TExt Representation).

## How It Works

UNITER (Chen et al., ECCV 2020) uses conditional masking pre-training where either
image regions or text tokens are masked, forcing the model to learn cross-modal alignment.
It uses a single-stream transformer for joint image-text encoding.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `UNITEROptions(UNITEROptions)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ImageMaskProbability` | Gets or sets the image region masking probability during training. |
| `MaxImageRegions` | Gets or sets the maximum number of image regions. |
| `TextMaskProbability` | Gets or sets the text token masking probability during training. |

