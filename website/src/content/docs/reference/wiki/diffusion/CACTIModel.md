---
title: "CACTIModel<T>"
description: "CACTI model for content-aware controllable text-to-image style transfer."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.StyleTransfer`

CACTI model for content-aware controllable text-to-image style transfer.

## For Beginners

CACTI is smart about style transfer — it knows which parts of
your image are important (like faces or text) and applies less style there to keep
them recognizable, while fully stylizing less important areas like backgrounds.

## How It Works

CACTI uses content-aware attention modulation to transfer style while preserving
semantic content regions. It identifies content-important areas and reduces style
transfer intensity there while fully stylizing background regions.

