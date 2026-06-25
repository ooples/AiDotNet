---
title: "RemoteCLIPOptions"
description: "Configuration options for the RemoteCLIP model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.VisionLanguage.Encoders`

Configuration options for the RemoteCLIP model.

## For Beginners

RemoteCLIP is CLIP trained specifically to understand satellite and
aerial images. It knows what "urban area", "agricultural land", "forest", "water body" etc.
look like from above, and can classify and search through satellite imagery using text queries.

## How It Works

RemoteCLIP (Liu et al., 2023) adapts CLIP for remote sensing imagery. It is fine-tuned on
curated remote sensing image-text datasets covering satellite imagery, aerial photography,
and geospatial data. It supports tasks like zero-shot scene classification, image-text
retrieval, and object counting in remote sensing contexts.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `RemoteCLIPOptions` | Initializes default RemoteCLIP options. |
| `RemoteCLIPOptions(RemoteCLIPOptions)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Domain` | Gets or sets the domain specialization. |
| `GroundSampleDistance` | Gets or sets the ground sample distance in meters per pixel. |
| `LossType` | Gets or sets the contrastive loss type. |
| `PromptTemplate` | Gets or sets the remote sensing specific text prompt template. |
| `UseMultiScale` | Gets or sets whether to use multi-scale features for remote sensing. |

