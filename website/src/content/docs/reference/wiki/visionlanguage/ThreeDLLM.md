---
title: "ThreeDLLM<T>"
description: "3D-LLM: injects 3D spatial features into large language models."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.ThreeD`

3D-LLM: injects 3D spatial features into large language models.

## For Beginners

3D-LLM is a vision-language model that enables language
understanding of 3D environments and scenes. Default values follow the original paper
settings.

## How It Works

3D-LLM (UCLA, 2023) injects 3D spatial features into large language models by rendering
multi-view images from 3D scenes and projecting 2D features back into 3D space. This
enables the LLM to understand spatial layouts, answer questions about 3D environments,
and follow language-guided navigation instructions within reconstructed 3D scenes.

**References:**

- Paper: "3D-LLM: Injecting the 3D World into Large Language Models (UCLA, 2023)"

## Methods

| Method | Summary |
|:-----|:--------|
| `GenerateFrom3D(Tensor<>,String)` | Processes 3D point cloud using 3D-LLM's multi-view feature lifting approach. |
| `GenerateFromImage(Tensor<>,String)` | Generates from 2D image using 3D-LLM's multi-view feature lifting approach. |

