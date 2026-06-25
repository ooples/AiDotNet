---
title: "SegFormerModelSize"
description: "Defines the model size variants for SegFormer (Mix Transformer backbone)."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums`

Defines the model size variants for SegFormer (Mix Transformer backbone).

## For Beginners

SegFormer comes in six sizes (B0 through B5). Smaller sizes (B0)
are faster and use less memory, while larger sizes (B5) are more accurate but require
more compute. B0 is a great starting point for experimentation, while B2-B3 offer
a good balance for production use.

## How It Works

**Technical Details:** Each size uses a different Mix Transformer (MiT) backbone
with varying embedding dimensions, transformer depths, and attention heads.

**Reference:** Xie et al., "SegFormer: Simple and Efficient Design for Semantic
Segmentation with Transformers", NeurIPS 2021.

## Fields

| Field | Summary |
|:-----|:--------|
| `B0` | MiT-B0: Smallest variant (3.8M params). |
| `B1` | MiT-B1: Small variant (13.7M params). |
| `B2` | MiT-B2: Medium variant (25.4M params). |
| `B3` | MiT-B3: Large variant (45.2M params). |
| `B4` | MiT-B4: Extra-large variant (62.0M params). |
| `B5` | MiT-B5: Largest variant (82.0M params). |

