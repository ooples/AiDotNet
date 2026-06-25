---
title: "ViLBERTOptions"
description: "Configuration options for ViLBERT (Vision-and-Language BERT) with co-attention."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.VisionLanguage.Foundational`

Configuration options for ViLBERT (Vision-and-Language BERT) with co-attention.

## How It Works

ViLBERT (Lu et al., NeurIPS 2019) processes images and text in two parallel BERT-like streams
connected by co-attention transformer layers, enabling rich cross-modal interaction.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ViLBERTOptions(ViLBERTOptions)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `MaxVisualRegions` | Gets or sets the maximum number of visual regions (object proposals). |
| `NumCoAttentionLayers` | Gets or sets the number of co-attention layers between streams. |

