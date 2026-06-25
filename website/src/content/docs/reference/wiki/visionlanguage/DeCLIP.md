---
title: "DeCLIP<T>"
description: "DeCLIP (Data-efficient CLIP) model using self-supervised and nearest-neighbor supervision for data efficiency."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.Encoders`

DeCLIP (Data-efficient CLIP) model using self-supervised and nearest-neighbor supervision for data efficiency.

## For Beginners

DeCLIP achieves CLIP-level performance with much less training
data by extracting extra supervision from the data itself. It combines contrastive image-text
matching with image self-supervision (SimSiam), text masked language modeling, and nearest-
neighbor supervision — squeezing more learning from each image-text pair. Default values
follow the original paper settings.

## How It Works

DeCLIP (Li et al., ICLR 2022) achieves CLIP-level performance with less data by combining contrastive
learning with image self-supervision (SimSiam), text MLM, and nearest-neighbor supervision.

**References:**

- Paper: "Supervision Exists Everywhere: A Data-Efficient Contrastive Language-Image Pre-training Paradigm" (Li et al., ICLR 2022)

## Methods

| Method | Summary |
|:-----|:--------|
| `GetExtraTrainableLayers` |  |

