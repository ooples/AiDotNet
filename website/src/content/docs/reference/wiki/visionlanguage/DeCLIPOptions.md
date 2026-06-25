---
title: "DeCLIPOptions"
description: "Configuration options for the DeCLIP (Data-efficient CLIP) model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.VisionLanguage.Encoders`

Configuration options for the DeCLIP (Data-efficient CLIP) model.

## For Beginners

DeCLIP learns more from less data by using multiple learning strategies
at once. While regular CLIP only learns from image-text pairs, DeCLIP also learns from images
alone (self-supervision) and from similar images nearby in the dataset. This makes it much more
data-efficient - achieving the same performance with fewer training examples.

## How It Works

DeCLIP (Li et al., ICLR 2022) improves CLIP's data efficiency by adding self-supervised learning
objectives alongside the contrastive image-text loss. It uses image self-supervision (SimSiam),
text self-supervision (masked language modeling), and nearest-neighbor supervision to extract
more learning signal from each image-text pair.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DeCLIPOptions` | Initializes default DeCLIP options. |
| `DeCLIPOptions(DeCLIPOptions)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ImageSelfSupervisedWeight` | Gets or sets the weight for the image self-supervised (SimSiam) loss. |
| `LossType` | Gets or sets the contrastive loss type. |
| `NearestNeighborWeight` | Gets or sets the weight for the nearest-neighbor supervision loss. |
| `NumNearestNeighbors` | Gets or sets the number of nearest neighbors for supervision. |
| `TextMLMWeight` | Gets or sets the weight for the text masked language modeling loss. |
| `TextMaskingRatio` | Gets or sets the masking ratio for text masked language modeling. |

