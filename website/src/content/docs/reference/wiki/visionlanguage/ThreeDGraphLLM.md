---
title: "ThreeDGraphLLM<T>"
description: "3DGraphLLM: 3D scene graph as LLM input for spatial reasoning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.ThreeD`

3DGraphLLM: 3D scene graph as LLM input for spatial reasoning.

## For Beginners

3DGraphLLM is a vision-language model that uses scene graphs
for structured 3D spatial reasoning. Default values follow the original paper settings.

## How It Works

3DGraphLLM (CogAI, 2025) uses 3D scene graphs as structured input for large language models.
It constructs semantic graphs from 3D scenes where nodes represent objects with their spatial
properties and edges encode spatial relationships, providing the LLM with a structured
representation for precise spatial reasoning, object counting, and relationship understanding.

**References:**

- Paper: "3DGraphLLM: 3D Scene Graph as Input for Large Language Models (CogAI, 2025)"

## Methods

| Method | Summary |
|:-----|:--------|
| `GenerateFrom3D(Tensor<>,String)` | Processes 3D point cloud using 3DGraphLLM's scene graph approach. |
| `GenerateFromImage(Tensor<>,String)` | Generates from 2D image using 3DGraphLLM's scene graph approach. |

