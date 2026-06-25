---
title: "KLoRAStyleModel<T>"
description: "K-LoRA Style model for composable style transfer by merging multiple LoRA style adapters."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.StyleTransfer`

K-LoRA Style model for composable style transfer by merging multiple LoRA style adapters.

## For Beginners

K-LoRA lets you mix multiple artistic styles. Each style is
a small add-on (LoRA adapter), and you can blend them — for example, 70% Van Gogh
colors with 30% Picasso geometry. This gives you creative control over the final look.

## How It Works

K-LoRA trains separate LoRA adapters for different style aspects (color, texture,
composition) and merges them with adjustable weights. This enables mixing multiple
styles with fine-grained control over each style's contribution.

