---
title: "SAM<T>"
description: "Segment Anything Model (SAM) vision encoder for promptable image segmentation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.Encoders`

Segment Anything Model (SAM) vision encoder for promptable image segmentation.

## For Beginners

SAM (Segment Anything Model) from Meta is a promptable
segmentation model — you give it an image and a prompt (a point click, a bounding box,
or a rough mask) and it segments the object you indicated. Trained on over 1 billion masks,
its ViT encoder uses windowed attention with occasional global attention for efficient
high-resolution processing. Default values follow the original paper settings.

## How It Works

SAM (Kirillov et al., 2023) consists of a ViT image encoder producing image embeddings, a prompt
encoder that handles points/boxes/masks, and a lightweight mask decoder. The image encoder uses
windowed attention with occasional global attention blocks for efficiency at high resolution.

**References:**

- Paper: "Segment Anything" (Kirillov et al., 2023)

