---
title: "WatermarkConfig"
description: "Configuration for watermarking modules."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Safety`

Configuration for watermarking modules.

## For Beginners

Watermarking embeds invisible markers in AI-generated content
so it can later be identified as AI-generated. This is increasingly required by
regulations like the EU AI Act (Article 50).

## How It Works

**References:**

- SynthID-Text: Production text watermarking at scale (Google DeepMind, Nature 2024)
- SynthID-Image: Internet-scale image watermarking (Google DeepMind, 2025)
- Only 38% of AI generators implement adequate watermarking (2025)

## Properties

| Property | Summary |
|:-----|:--------|
| `AudioWatermarking` | Gets or sets whether audio watermarking is enabled. |
| `DetectionMode` | Gets or sets watermark detection mode (detect existing watermarks on input). |
| `ImageWatermarking` | Gets or sets whether image watermarking is enabled. |
| `TextWatermarking` | Gets or sets whether text watermarking is enabled. |
| `WatermarkStrength` | Gets or sets the watermark strength (0-1). |

