---
title: "MedCLIPOptions"
description: "Configuration options for the MedCLIP model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.VisionLanguage.Encoders`

Configuration options for the MedCLIP model.

## For Beginners

MedCLIP is designed for medical imaging but with a clever twist: instead
of needing perfectly matched pairs of images and descriptions, it can learn from any image-text
combination that describes the same medical condition. This greatly increases the amount of
usable training data in the medical domain.

## How It Works

MedCLIP (Wang et al., 2022) from UCSD addresses the challenge of limited medical image-text pairs
by decoupling image and text inputs during contrastive learning. Instead of requiring exact
image-text pairs, it uses a semantic matching loss that allows any image to be paired with any
text description that shares the same medical concepts (e.g., diagnosis, anatomy).

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MedCLIPOptions` | Initializes default MedCLIP options. |
| `MedCLIPOptions(MedCLIPOptions)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Domain` | Gets or sets the domain specialization. |
| `EntitySimilarityThreshold` | Gets or sets the medical entity similarity threshold for soft labeling. |
| `LossType` | Gets or sets the contrastive loss type. |
| `SemanticMatchingWeight` | Gets or sets the weight for the semantic matching loss. |
| `UseEntityExtraction` | Gets or sets whether to use entity extraction for concept alignment. |
| `VisionBackbone` | Gets or sets the vision backbone used by MedCLIP. |

