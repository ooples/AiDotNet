---
title: "InstantStyleModel<T>"
description: "InstantStyle model for zero-shot style transfer using IP-Adapter with style-content separation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.StyleTransfer`

InstantStyle model for zero-shot style transfer using IP-Adapter with style-content separation.

## For Beginners

InstantStyle transfers artistic styles instantly — no training
or fine-tuning needed per style. It smartly injects style information into specific
parts of the model to capture the look without copying the content of the style image.

## How It Works

InstantStyle achieves high-quality style transfer in a single forward pass by
selectively injecting IP-Adapter image features into only the style-relevant
attention layers, preventing content leakage from the style reference.

Reference: Wang et al., "InstantStyle: Free Lunch towards Style-Preserving in Text-to-Image Generation", 2024

