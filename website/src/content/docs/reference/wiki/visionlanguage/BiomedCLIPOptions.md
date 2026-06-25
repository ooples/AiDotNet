---
title: "BiomedCLIPOptions"
description: "Configuration options for the BiomedCLIP model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.VisionLanguage.Encoders`

Configuration options for the BiomedCLIP model.

## For Beginners

BiomedCLIP is a version of CLIP trained specifically on medical and
biological images and their descriptions from scientific papers. This means it understands
medical images (X-rays, microscopy, etc.) much better than general-purpose CLIP.

## How It Works

BiomedCLIP (Zhang et al., 2023) from Microsoft is a CLIP model fine-tuned on PMC-15M, a dataset
of 15 million biomedical image-text pairs from PubMed Central. It uses a ViT-B/16 vision encoder
and PubMedBERT text encoder, achieving state-of-the-art zero-shot biomedical image classification.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `BiomedCLIPOptions` | Initializes default BiomedCLIP options. |
| `BiomedCLIPOptions(BiomedCLIPOptions)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Dataset` | Gets or sets the pre-training dataset. |
| `Domain` | Gets or sets the domain specialization. |
| `LossType` | Gets or sets the contrastive loss type. |
| `MedicalTextEncoder` | Gets or sets the medical text encoder variant. |
| `UseBiomedicalAugmentations` | Gets or sets whether to use domain-specific image augmentations. |

