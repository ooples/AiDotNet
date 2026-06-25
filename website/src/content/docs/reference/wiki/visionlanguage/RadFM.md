---
title: "RadFM<T>"
description: "RadFM: 3D ViT with perceiver for radiology report generation and VQA."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.Medical`

RadFM: 3D ViT with perceiver for radiology report generation and VQA.

## For Beginners

RadFM is a vision-language model for radiology report generation
and visual question answering on medical imaging data. Default values follow the original
paper settings.

## How It Works

RadFM (2024) is a generalist foundation model for radiology that uses a 3D Vision Transformer
with a perceiver module to encode volumetric CT and MRI data. It handles both 2D radiographs
and 3D volumetric scans, generating radiology reports, answering visual questions about
medical images, and supporting multi-modal clinical decision-making.

**References:**

- Paper: "RadFM: Towards Generalist Foundation Model for Radiology (Various, 2024)"

## Methods

| Method | Summary |
|:-----|:--------|
| `GenerateFromImage(Tensor<>,String)` | Generates text from a radiology image using RadFM's 3D-aware perceiver pipeline. |

