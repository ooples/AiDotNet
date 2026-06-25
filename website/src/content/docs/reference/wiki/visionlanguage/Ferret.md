---
title: "Ferret<T>"
description: "Ferret: spatial-aware visual sampler for free-form region referring and grounding."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.Grounding`

Ferret: spatial-aware visual sampler for free-form region referring and grounding.

## For Beginners

Ferret is a vision-language model that can locate objects in images
based on natural language descriptions and refer to regions of any shape. Default values follow
the original paper settings.

## How It Works

Ferret (You et al., 2023) introduces a spatial-aware visual sampler that can refer to and
ground anything at any granularity. It uses a hybrid region representation combining discrete
coordinates with continuous visual features and a spatial-aware visual sampler to handle
free-form referring regions of any shape including points, boxes, and scribbles.

**References:**

- Paper: "Ferret: Refer and Ground Anything Anywhere at Any Granularity" (Apple, 2023)

## Methods

| Method | Summary |
|:-----|:--------|
| `GroundText(Tensor<>,String)` | Grounds text using Ferret's spatial-aware visual sampler for free-form referring. |

