---
title: "ConsisLoRAModel<T>"
description: "ConsisLoRA model for consistent style transfer using LoRA with content-style disentanglement."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.StyleTransfer`

ConsisLoRA model for consistent style transfer using LoRA with content-style disentanglement.

## For Beginners

ConsisLoRA creates reliable style adapters that transfer only
the style (colors, brushstrokes, textures) without changing the content. Multiple images
generated with the same ConsisLoRA will have a consistent visual style.

## How It Works

ConsisLoRA improves LoRA-based style transfer by explicitly disentangling content
and style information during LoRA training. Uses consistency regularization to ensure
style features are captured in the LoRA weights while content features pass through.

