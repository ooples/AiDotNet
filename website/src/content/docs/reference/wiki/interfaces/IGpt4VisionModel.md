---
title: "IGpt4VisionModel<T>"
description: "Defines the contract for GPT-4V-style models that combine vision understanding with large language model capabilities."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Defines the contract for GPT-4V-style models that combine vision understanding with large language model capabilities.

## For Beginners

GPT-4V is like giving ChatGPT the ability to see!

Key capabilities:

- Visual reasoning: Understanding relationships, counting, spatial awareness
- Multi-turn dialogue: Context-aware conversations about images
- Document understanding: Reading and analyzing documents, charts, diagrams
- Code generation from screenshots: Understanding UI and generating code
- Creative tasks: Describing images poetically, writing stories from images

Architecture concepts:

1. Vision Encoder: Processes images into visual tokens
2. Visual-Language Alignment: Maps visual features to LLM embedding space
3. Large Language Model: Generates text responses conditioned on visual input
4. Multi-modal Attention: Allows text to attend to relevant image regions

## How It Works

GPT-4V represents the integration of vision capabilities into large language models,
enabling sophisticated visual reasoning, multi-turn conversations about images,
and complex visual-linguistic tasks.

## Properties

| Property | Summary |
|:-----|:--------|
| `ContextWindowSize` | Gets the context window size in tokens. |
| `MaxImageResolution` | Gets the maximum resolution supported for input images. |
| `MaxImagesPerRequest` | Gets the maximum number of images that can be processed in a single request. |
| `SupportedDetailLevels` | Gets the supported image detail levels. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AnalyzeChart(Tensor<>)` | Analyzes a chart or graph and extracts data. |
| `AnalyzeDocument(Tensor<>,String,String)` | Analyzes a document image (PDF page, screenshot, etc.). |
| `AnswerVisualQuestion(Tensor<>,String)` | Answers a visual question with confidence score. |
| `Chat(Tensor<>,IEnumerable<ValueTuple<String,String>>,String,Int32)` | Conducts a multi-turn conversation about an image. |
| `CompareImages(Tensor<>,Tensor<>,String)` | Compares two images and describes their differences. |
| `DescribeImage(Tensor<>,String,String)` | Describes an image with specified style and detail level. |
| `DetectObjects(Tensor<>,String)` | Identifies and locates objects in an image with bounding boxes. |
| `EvaluateImageQuality(Tensor<>)` | Evaluates image quality and provides improvement suggestions. |
| `ExtractStructuredData(Tensor<>,String)` | Extracts structured data from an image. |
| `ExtractText(Tensor<>,Boolean)` | Performs OCR with layout understanding. |
| `Generate(Tensor<>,String,Int32,Double)` | Generates a response based on an image and text prompt. |
| `GenerateCodeFromUI(Tensor<>,String,String)` | Generates code from a UI screenshot. |
| `GenerateEditInstructions(Tensor<>,String)` | Generates image editing instructions based on a modification request. |
| `GenerateFromMultipleImages(IEnumerable<Tensor<>>,String,Int32,Double)` | Generates a response based on multiple images and text prompt. |
| `GenerateStory(Tensor<>,String,String)` | Generates a creative story or narrative based on an image. |
| `GetAttentionMap(Tensor<>,String)` | Gets attention weights showing which image regions influenced the response. |
| `SafetyCheck(Tensor<>)` | Identifies potential safety concerns in an image. |
| `VisualReasoning(Tensor<>,String,String)` | Performs visual reasoning tasks. |

