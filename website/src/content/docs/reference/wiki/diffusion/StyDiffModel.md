---
title: "StyDiffModel<T>"
description: "StyDiff model for diffusion-based artistic style transfer with content preservation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.StyleTransfer`

StyDiff model for diffusion-based artistic style transfer with content preservation.

## For Beginners

StyDiff takes the artistic style from one image (like a painting)
and applies it to another image (like a photo) while keeping the photo's content intact.
The result looks like the original photo painted in the reference style.

## How It Works

StyDiff performs style transfer by injecting style features from a reference image
into the diffusion process via cross-attention. Content structure is preserved through
DDIM inversion while the style is transferred from the reference.

