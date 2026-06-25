---
title: "IReferringSegmentation<T>"
description: "Interface for referring segmentation models that segment objects based on natural language descriptions, including complex reasoning about spatial relationships and attributes."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Interface for referring segmentation models that segment objects based on natural language
descriptions, including complex reasoning about spatial relationships and attributes.

## For Beginners

Referring segmentation lets you describe exactly what you want
using natural language — even complex descriptions.

Examples of what you can ask:

- "The person standing behind the counter" (spatial reasoning)
- "The animal that could be dangerous" (world knowledge / reasoning)
- "The object that doesn't belong in this kitchen" (contextual reasoning)
- "Track the person in the red shirt throughout the video" (video + language)

This is more powerful than open-vocabulary segmentation because it understands context
and relationships, not just category names.

Models implementing this interface:

- LISA (CVPR 2024 Oral, LLaVA + SAM reasoning segmentation)
- VideoLISA (NeurIPS 2024, video reasoning segmentation)
- GLaMM (CVPR 2024, grounded language model)
- OMG-LLaVA (NeurIPS 2024, pixel-level reasoning)
- PixelLM (CVPR 2024, segmentation codebook)

## How It Works

Referring segmentation goes beyond open-vocabulary segmentation by supporting complex
natural language queries that require reasoning about object attributes, spatial relationships,
and even world knowledge. These models typically use large language models (LLMs) to
understand complex queries.

## Properties

| Property | Summary |
|:-----|:--------|
| `MaxTextLength` | Gets the maximum input text length in tokens. |
| `SupportsConversation` | Gets whether this model supports multi-turn conversation. |
| `SupportsVideoInput` | Gets whether this model supports video input. |

## Methods

| Method | Summary |
|:-----|:--------|
| `SegmentFromConversation(Tensor<>,IReadOnlyList<ValueTuple<String,String>>,String)` | Segments objects from a multi-turn conversation context. |
| `SegmentFromExpression(Tensor<>,String)` | Segments objects described by a natural language expression. |
| `SegmentVideoFromExpression(Tensor<>,String)` | Segments objects in a video based on a natural language expression. |

