---
title: "VideoModelVariant"
description: "Defines common model size variants for video processing models (SR, interpolation, flow, etc.)."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums`

Defines common model size variants for video processing models (SR, interpolation, flow, etc.).

## For Beginners

Think of these like clothing sizes. "Tiny" is the fastest but least
accurate, "Base" is the recommended starting point, and "Large"/"XLarge" are for when
you need maximum quality and have powerful hardware.

## How It Works

Video models typically come in multiple sizes trading off speed vs quality:

- Smaller variants (Tiny, Small) run faster with lower memory, suitable for real-time applications
- Larger variants (Large, XLarge) produce higher quality but require more compute
- Base is the default recommended configuration balancing speed and quality

## Fields

| Field | Summary |
|:-----|:--------|
| `Base` | Base variant: default recommended configuration. |
| `Large` | Large variant: increased capacity for higher quality. |
| `Pro` | Pro variant: production-optimized variant with enhanced features. |
| `Small` | Small variant: reduced size for faster inference (~20 FPS). |
| `Tiny` | Tiny variant: minimal size for maximum speed (~30+ FPS). |
| `XLarge` | Extra-large variant: maximum capacity. |

