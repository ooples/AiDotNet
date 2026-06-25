---
title: "ILLaVAModel<T>"
description: "Defines the contract for LLaVA (Large Language and Vision Assistant) models."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Defines the contract for LLaVA (Large Language and Vision Assistant) models.

## For Beginners

LLaVA is like giving eyes to ChatGPT!

Architecture:

1. Vision Encoder (CLIP ViT): Converts images to feature vectors
2. Projection Layer: Maps visual features to LLM's text embedding space
3. Large Language Model (LLaMA/Vicuna): Generates responses

Key capabilities:

- Visual conversations: "What's in this image?" followed by "What color is the car?"
- Visual reasoning: Understanding relationships, counting, spatial awareness
- Instruction following: "Describe this image as if you were a poet"
- Multi-turn dialogue: Context-aware conversations about images

Why LLaVA is popular:

- Simple but effective architecture
- Open-source and reproducible
- Strong performance on visual understanding benchmarks
- Efficient training with visual instruction tuning

## How It Works

LLaVA connects a vision encoder (like CLIP ViT) with a large language model (like LLaMA/Vicuna)
through a simple projection layer, enabling visual instruction-following and conversational AI
about images.

## Properties

| Property | Summary |
|:-----|:--------|
| `LanguageModelBackbone` | Gets the language model backbone used for generation. |
| `NumVisualTokens` | Gets the maximum number of visual tokens used per image. |
| `VisionEncoderType` | Gets the vision encoder type. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Chat(Tensor<>,IEnumerable<ValueTuple<String,String>>,String,Int32,Double)` | Continues a multi-turn conversation about an image. |
| `CompareImages(Tensor<>,Tensor<>,IEnumerable<String>)` | Compares two images and describes their differences. |
| `DescribeRegions(Tensor<>,IEnumerable<Vector<>>)` | Generates a detailed description of specific regions in an image. |
| `ExtractVisualFeatures(Tensor<>)` | Extracts visual features before projection to LLM space. |
| `Generate(Tensor<>,String,Int32,Double,Double)` | Generates a response to a text prompt about an image. |
| `GenerateMultiple(Tensor<>,String,Int32,Double)` | Generates multiple diverse responses for the same prompt. |
| `GroundObject(Tensor<>,String)` | Performs visual grounding to locate objects described by text. |
| `ProjectToLanguageSpace(Tensor<>)` | Projects visual features to the LLM's embedding space. |

