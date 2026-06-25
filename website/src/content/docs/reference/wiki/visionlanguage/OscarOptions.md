---
title: "OscarOptions"
description: "Configuration options for Oscar (Object-Semantics Aligned pre-training)."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.VisionLanguage.Foundational`

Configuration options for Oscar (Object-Semantics Aligned pre-training).

## How It Works

Oscar (Li et al., ECCV 2020) uses detected object tags as "anchor points" to align image
regions with text tokens, forming triples of (word tokens, object tags, region features)
that are fed into a single BERT transformer.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `OscarOptions(OscarOptions)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ContrastiveLossWeight` | Gets or sets the contrastive loss weight for tag-text alignment. |
| `MaxImageRegions` | Gets or sets the maximum number of image regions. |
| `MaxObjectTags` | Gets or sets the maximum number of object tags per image. |

