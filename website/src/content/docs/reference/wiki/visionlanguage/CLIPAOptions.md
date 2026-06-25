---
title: "CLIPAOptions"
description: "Configuration options for the CLIPA (CLIP with Inverse scaling law and Accelerated training) model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.VisionLanguage.Encoders`

Configuration options for the CLIPA (CLIP with Inverse scaling law and Accelerated training) model.

## For Beginners

CLIPA finds that you can train CLIP much faster by using lower resolution
images and shorter text during most of the training, then switching to full resolution at the end.
This is like studying cliff notes first to get the big picture, then reading the full text for details.

## How It Works

CLIPA (Li et al., 2023) discovers an "inverse scaling law" for CLIP training: using shorter
image sequences and text lengths during the bulk of training, then fine-tuning at full resolution.
This reduces training cost by 7-8x while maintaining performance, enabling efficient scaling.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CLIPAOptions` | Initializes default CLIPA options. |
| `CLIPAOptions(CLIPAOptions)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `InitialImageSize` | Gets or sets the initial (reduced) image size for the bulk of training. |
| `InitialSequenceLength` | Gets or sets the initial (reduced) text sequence length for the bulk of training. |
| `IsFineTuningPhase` | Gets or sets whether the model is in the fine-tuning phase (full resolution). |
| `LossType` | Gets or sets the contrastive loss type. |
| `ReducedResolutionFraction` | Gets or sets the fraction of training done at reduced resolution before switching to full. |

