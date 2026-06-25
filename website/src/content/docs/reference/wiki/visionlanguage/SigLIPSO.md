---
title: "SigLIPSO<T>"
description: "SigLIP-SO (Shape-Optimized SigLIP) 400M vision encoder widely used as a VLM backbone."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.Encoders`

SigLIP-SO (Shape-Optimized SigLIP) 400M vision encoder widely used as a VLM backbone.

## For Beginners

SigLIP-SO is a shape-optimized version of SigLIP that carefully
tunes the width and depth of the Vision Transformer for the 400M parameter budget. The
resulting SO-400M/14 model has become one of the most popular vision backbones, used by
LLaVA, PaliGemma, and many other multimodal models as their vision encoder. Default
values follow the original paper settings.

## How It Works

SigLIP-SO (Zhai et al., 2023) optimizes the ViT width/depth ratio for the 400M parameter budget.
The resulting SO-400M/14 model produces high-quality visual features adopted by LLaVA, PaliGemma,
and other popular VLMs. It uses sigmoid contrastive loss for training.

**References:**

- Paper: "Sigmoid Loss for Language Image Pre-Training" (Zhai et al., ICCV 2023)

