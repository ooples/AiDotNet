---
title: "VisualBERTOptions"
description: "Configuration options for VisualBERT single-stream fusion model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.VisionLanguage.Foundational`

Configuration options for VisualBERT single-stream fusion model.

## How It Works

VisualBERT (Li et al., 2019) concatenates visual tokens (from Faster R-CNN) with text tokens
in a single BERT transformer stream, allowing implicit cross-modal alignment through self-attention.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `VisualBERTOptions(VisualBERTOptions)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `MaxVisualTokens` | Gets or sets the maximum number of visual tokens. |
| `UseVisualSegmentEmbeddings` | Gets or sets whether to use visual segment embeddings. |

