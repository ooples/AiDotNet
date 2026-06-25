---
title: "StyleAlignedEditModel<T>"
description: "StyleAligned-Edit model for consistent multi-image style editing via shared self-attention."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.StyleTransfer`

StyleAligned-Edit model for consistent multi-image style editing via shared self-attention.

## For Beginners

When you need to edit multiple images with the same visual style
(like a series of product photos or a consistent set of illustrations), StyleAligned-Edit
ensures they all look like they belong together by sharing style information across images.

## How It Works

StyleAligned-Edit extends the StyleAligned shared attention mechanism to image editing,
ensuring that multiple edited images maintain visual consistency. All images in a batch
share self-attention features, enforcing a coherent visual style across the set.

Reference: Hertz et al., "Style Aligned Image Generation via Shared Attention", CVPR 2024

