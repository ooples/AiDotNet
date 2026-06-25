---
title: "IVisualGroundingModel<T>"
description: "Interface for visual grounding models that localize objects or regions from natural language descriptions."
section: "API Reference"
---

`Interfaces` · `AiDotNet.VisionLanguage.Interfaces`

Interface for visual grounding models that localize objects or regions from natural language descriptions.

## How It Works

Visual grounding models take an image and a text query and produce bounding boxes,
segmentation masks, or region proposals for the objects/regions described by the text.
Architectures include:

- Grounding DINO: DINO detector + grounded pre-training for open-set detection
- GLaMM: Pixel grounding with LMM generating text + segmentation masks
- Ferret: Spatial-aware visual sampler for free-form region inputs
- OWL-ViT/OWLv2: Open-vocabulary detection via CLIP-aligned ViT

## Properties

| Property | Summary |
|:-----|:--------|
| `MaxDetections` | Gets the maximum number of detections the model can produce per image. |

## Methods

| Method | Summary |
|:-----|:--------|
| `DetectObjects(Tensor<>,IReadOnlyList<String>)` | Detects objects in the image matching a set of category descriptions. |
| `GroundText(Tensor<>,String)` | Grounds a text query in the image, producing bounding box coordinates. |

