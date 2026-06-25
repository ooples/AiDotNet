---
title: "DINOv2<T>"
description: "DINOv2 self-supervised vision encoder producing universal visual features without labels."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.Encoders`

DINOv2 self-supervised vision encoder producing universal visual features without labels.

## For Beginners

DINOv2 from Meta learns powerful visual features without any
labeled data using self-supervised training on 142 million curated images. Its features
are so general that a simple linear classifier on top achieves strong results for image
classification, segmentation, and depth estimation — making it a universal vision backbone.
Default values follow the original paper settings.

## How It Works

DINOv2 (Oquab et al., 2024) trains ViT with iBOT masked image modeling + DINO self-distillation
on LVD-142M curated images. Register tokens reduce artifact patterns in attention maps. The model
produces features usable for classification, segmentation, and depth estimation via linear probing.

**References:**

- Paper: "DINOv2: Learning Robust Visual Features without Supervision" (Oquab et al., 2024)

