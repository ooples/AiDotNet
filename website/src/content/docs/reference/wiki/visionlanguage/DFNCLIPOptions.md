---
title: "DFNCLIPOptions"
description: "Configuration options for the DFN-CLIP (Data Filtering Networks for CLIP) model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.VisionLanguage.Encoders`

Configuration options for the DFN-CLIP (Data Filtering Networks for CLIP) model.

## For Beginners

DFN-CLIP uses a clever bootstrapping trick: first train a small model,
then use that model to find the best training data from a huge noisy pool, and finally train
a bigger model on just the good data. It's like having a teacher pre-screen study materials.

## How It Works

DFN-CLIP (Fang et al., 2023) from Apple uses a small CLIP model as a "data filtering network"
to score and select high-quality image-text pairs from a large noisy pool. The filtered data
is then used to train a larger CLIP model, achieving 83.0% zero-shot on ImageNet with ViT-H/14.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DFNCLIPOptions` | Initializes default DFN-CLIP options. |
| `DFNCLIPOptions(DFNCLIPOptions)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Dataset` | Gets or sets the pre-training dataset. |
| `FilteredDatasetSizeMillions` | Gets or sets the target dataset size after filtering (in millions). |
| `FilteringThreshold` | Gets or sets the filtering threshold score for data selection. |
| `LossType` | Gets or sets the contrastive loss type. |

