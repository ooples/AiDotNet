---
title: "LXMERTOptions"
description: "Configuration options for LXMERT cross-modal encoder."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.VisionLanguage.Foundational`

Configuration options for LXMERT cross-modal encoder.

## How It Works

LXMERT (Tan and Bansal, EMNLP 2019) has three encoder types: object relationship encoder,
language encoder, and cross-modality encoder with cross-attention layers.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LXMERTOptions(LXMERTOptions)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `MaxVisualObjects` | Gets or sets the maximum number of visual objects. |
| `NumCrossModalityLayers` | Gets or sets the number of cross-modality encoder layers. |
| `NumRelationshipLayers` | Gets or sets the number of object relationship encoder layers. |

