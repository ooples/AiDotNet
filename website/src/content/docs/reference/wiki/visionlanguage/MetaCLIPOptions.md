---
title: "MetaCLIPOptions"
description: "Configuration options for the MetaCLIP model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.VisionLanguage.Encoders`

Configuration options for the MetaCLIP model.

## For Beginners

MetaCLIP focuses on training data quality rather than model architecture.
It carefully selects and balances the image-text pairs used for training, making sure the model
sees a diverse and representative set of concepts. Better data leads to a better model.

## How It Works

MetaCLIP (Xu et al., 2023) improves CLIP training through metadata-driven data curation.
Instead of using raw web-scraped data, MetaCLIP balances the training distribution by using
metadata from WordNet and Wikipedia to ensure diverse, high-quality image-text pairs. This
data-centric approach achieves 70.8% zero-shot on ImageNet with ViT-B/16 on 400M pairs.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MetaCLIPOptions` | Initializes default MetaCLIP options. |
| `MetaCLIPOptions(MetaCLIPOptions)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Dataset` | Gets or sets the pre-training dataset. |
| `LossType` | Gets or sets the contrastive loss type. |
| `MaxEntriesPerConcept` | Gets or sets the maximum number of entries per metadata concept for data balancing. |
| `UseSubStringMatching` | Gets or sets whether to use sub-string matching for metadata alignment. |

