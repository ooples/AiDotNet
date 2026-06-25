---
title: "SASTDModel<T>"
description: "SASTD model for structure-aware style transfer via diffusion with edge-guided generation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.StyleTransfer`

SASTD model for structure-aware style transfer via diffusion with edge-guided generation.

## For Beginners

SASTD preserves the edges and outlines of your image during
style transfer. This means buildings keep their shapes, faces keep their features,
and objects stay recognizable even with heavy stylization.

## How It Works

SASTD uses edge maps extracted from the content image to guide style transfer,
ensuring structural boundaries are preserved during stylization. The edge guidance
acts as an additional conditioning signal alongside the style reference.

