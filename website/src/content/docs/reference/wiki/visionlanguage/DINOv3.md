---
title: "DINOv3<T>"
description: "DINOv3 self-supervised vision encoder scaled to 7B parameters on 1.7B images."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.Encoders`

DINOv3 self-supervised vision encoder scaled to 7B parameters on 1.7B images.

## For Beginners

DINOv3 from Meta scales self-supervised vision training to
7 billion parameters on 1.7 billion images, outperforming even contrastive models like
SigLIP 2 on most vision benchmarks. It uses improved training recipes with SwiGLU feed-
forward networks and enhanced data augmentation, producing the strongest open-source
self-supervised vision encoder available. Default values follow the original paper
settings.

## How It Works

DINOv3 (Meta, 2025) scales self-supervised ViT training to 7B parameters on 1.7B images,
outperforming SigLIP 2 on most vision benchmarks. It uses improved training recipes with
SwiGLU FFN, enhanced data augmentation, and longer training schedules.

**References:**

- Paper: "DINOv3: Self-Supervised Vision at Scale" (Meta, 2025)

