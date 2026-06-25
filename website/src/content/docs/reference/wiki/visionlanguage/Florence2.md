---
title: "Florence2<T>"
description: "Florence-2 unified vision foundation model for captioning, detection, grounding, and OCR."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.Encoders`

Florence-2 unified vision foundation model for captioning, detection, grounding, and OCR.

## For Beginners

Florence-2 from Microsoft is a lightweight vision model
(0.23B-0.77B parameters) that handles many tasks through text prompts — captioning,
object detection, grounding, OCR, and segmentation — all in a single unified model.
It uses DaViT (Dual Attention ViT) as its vision encoder and generates structured
text output for each task. Default values follow the original paper settings.

## How It Works

Florence-2 (Xiao et al., Microsoft 2024) is a lightweight sequence-to-sequence vision model
(0.23B-0.77B) handling multiple tasks through a unified prompt-based approach. It uses DaViT
(Dual Attention ViT) as the vision encoder and a multi-task decoder for captioning, detection,
grounding, OCR, and segmentation.

**References:**

- Paper: "Florence-2: Advancing a Unified Representation for a Variety of Vision Tasks" (Xiao et al., 2024)

