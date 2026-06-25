---
title: "IFlamingoModel<T>"
description: "Defines the contract for Flamingo-style models with in-context visual learning capabilities."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Defines the contract for Flamingo-style models with in-context visual learning capabilities.

## For Beginners

Flamingo learns new visual tasks from examples you show it!

Key innovation - In-context learning:

- Show Flamingo a few example image-text pairs
- It learns the pattern from these examples
- Apply the pattern to new images WITHOUT any training

Architecture:

1. Vision Encoder: Extracts image features (Perceiver Resampler)
2. Gated Cross-Attention: Injects visual info into language model
3. Frozen LLM: Chinchilla-based language model

Example use case:

- Show 3 examples: [image1] "A red apple" [image2] "A blue car" [image3] "A green tree"
- Ask about new image: [image4] "What color?"
- Flamingo learns from examples that you want the color, answers correctly!

Why Flamingo is revolutionary:

- No fine-tuning needed for new tasks
- Adapts to new visual concepts on-the-fly
- Strong performance with minimal examples

## How It Works

Flamingo is a visual language model that excels at few-shot learning - it can learn new tasks
from just a few examples provided in the context. It uses gated cross-attention layers
interleaved with frozen LLM layers to integrate visual information.

## Properties

| Property | Summary |
|:-----|:--------|
| `LanguageModelBackbone` | Gets the language model backbone used for generation. |
| `MaxImagesInContext` | Gets the maximum number of images that can be processed in a single context. |
| `NumPerceiverTokens` | Gets the number of visual tokens per image after the Perceiver Resampler. |

## Methods

| Method | Summary |
|:-----|:--------|
| `DescribeVideo(IEnumerable<Tensor<>>,String,Int32)` | Generates captions for a video represented as a sequence of frames. |
| `ExtractPerceiverFeatures(Tensor<>)` | Extracts visual features using the Perceiver Resampler. |
| `FewShotGenerate(IEnumerable<ValueTuple<Tensor<>,String>>,Tensor<>,String,Int32)` | Performs few-shot visual learning with interleaved image-text examples. |
| `FewShotImageRetrieval(IEnumerable<Tensor<>>,String,IEnumerable<Tensor<>>,Int32)` | Retrieves the most similar images from a database using few-shot context. |
| `FewShotVQA(IEnumerable<ValueTuple<Tensor<>,String,String>>,Tensor<>,String)` | Performs visual question answering with few-shot examples. |
| `GenerateWithMultipleImages(IEnumerable<Tensor<>>,String,Int32)` | Generates text for multiple images interleaved in a single context. |
| `InContextClassify(IEnumerable<ValueTuple<Tensor<>,String>>,Tensor<>)` | Performs in-context visual classification without explicit labels. |
| `ScoreImageText(Tensor<>,String)` | Computes the log probability of a given text completion for an image. |

