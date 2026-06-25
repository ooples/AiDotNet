---
title: "FerretV2<T>"
description: "Ferret-v2: improved referring and grounding with enhanced spatial understanding."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.Grounding`

Ferret-v2: improved referring and grounding with enhanced spatial understanding.

## For Beginners

Ferret-v2 is an improved vision-language model for referring and grounding
that handles images at multiple resolutions for better spatial understanding. Default values follow
the original paper settings.

## How It Works

Ferret-v2 (Zhang et al., 2024) improves upon Ferret with multi-granularity visual encoding
that processes images at multiple resolutions, enhanced spatial understanding through DINOv2
features, and a flexible any-resolution scaling approach for handling diverse aspect ratios
and input sizes. The multi-granularity design captures both fine-grained details and global context.

**References:**

- Paper: "Ferret-v2: An Improved Baseline for Referring and Grounding with Multi-Granularity Visual Encoding" (Apple, 2024)

## Methods

| Method | Summary |
|:-----|:--------|
| `GroundText(Tensor<>,String)` | Grounds text using Ferret-v2's any-resolution high-res grounding approach. |

