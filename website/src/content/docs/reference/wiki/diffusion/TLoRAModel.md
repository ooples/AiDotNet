---
title: "TLoRAModel<T>"
description: "T-LoRA model for temporal LoRA-based style transfer with video-consistent stylization."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.StyleTransfer`

T-LoRA model for temporal LoRA-based style transfer with video-consistent stylization.

## For Beginners

T-LoRA applies artistic styles to videos without flickering.
Normal style transfer applied frame-by-frame causes shimmer and inconsistency.
T-LoRA ensures each frame has the same style applied smoothly, making the video
look like it was painted by the same artist throughout.

## How It Works

T-LoRA extends LoRA-based style transfer to video by incorporating temporal consistency
constraints. Style is applied uniformly across frames while respecting temporal coherence,
preventing flickering artifacts common in frame-by-frame stylization.

