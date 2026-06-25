---
title: "BridgeTowerOptions"
description: "Configuration options for BridgeTower cross-modal alignment model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.VisionLanguage.Foundational`

Configuration options for BridgeTower cross-modal alignment model.

## How It Works

BridgeTower (Xu et al., AAAI 2023) introduces bridge layers that connect vision and text
encoder layers at multiple levels, enabling fine-grained cross-modal alignment. Each bridge
layer consists of cross-attention between corresponding encoder layers.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `BridgeTowerOptions(BridgeTowerOptions)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `BridgeDim` | Gets or sets the bridge layer hidden dimension. |
| `NumBridgeLayers` | Gets or sets the number of bridge connection layers. |
| `UseBidirectionalBridges` | Gets or sets whether to use bi-directional bridges. |

