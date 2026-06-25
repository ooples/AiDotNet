---
title: "InternImageModelSize"
description: "Defines the model size variants for InternImage (DCNv3-based CNN backbone)."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums`

Defines the model size variants for InternImage (DCNv3-based CNN backbone).

## For Beginners

InternImage comes in five sizes (Tiny through Huge). It proves that
CNNs can match or exceed Vision Transformers when using modern deformable convolutions.
Tiny is great for quick experiments, while XL and Huge target maximum accuracy.

## How It Works

**Technical Details:** Each size uses a different DCNv3 backbone configuration with
varying channel widths, depths, and group sizes. All variants use the UPerNet decoder.

**Reference:** Wang et al., "InternImage: Exploring Large-Scale Vision Foundation
Models with Deformable Convolutions", CVPR 2023.

## Fields

| Field | Summary |
|:-----|:--------|
| `Base` | InternImage-B: Base variant (97M params). |
| `Huge` | InternImage-H: Huge variant (1.08B params). |
| `Small` | InternImage-S: Small variant (50M params). |
| `Tiny` | InternImage-T: Tiny variant (30M params). |
| `XL` | InternImage-XL: Extra-large variant (335M params). |

