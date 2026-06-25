---
title: "OWLViT<T>"
description: "OWL-ViT: open-vocabulary object detection via ViT + CLIP alignment."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.Grounding`

OWL-ViT: open-vocabulary object detection via ViT + CLIP alignment.

## For Beginners

OWL-ViT is a vision-language model that detects objects using
text descriptions by leveraging CLIP's vision-language alignment at the patch level. Default
values follow the original paper settings.

## How It Works

OWL-ViT (Minderer et al., 2022) repurposes a CLIP ViT image encoder for open-vocabulary
detection by treating each patch token as a candidate object. Per-patch MLP heads predict
bounding boxes, while class scores come from cosine similarity between patch tokens and
CLIP text embeddings of category names. This is a query-free approach where every ViT
patch serves as a candidate detection, eliminating the need for learned object queries.

**References:**

- Paper: "Simple Open-Vocabulary Object Detection with Vision Transformers" (Google, 2022)

## Methods

| Method | Summary |
|:-----|:--------|
| `GroundText(Tensor<>,String)` | Grounds text using OWL-ViT's CLIP-aligned patch-level detection approach. |

