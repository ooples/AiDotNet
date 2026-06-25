---
title: "IBlip2Model<T>"
description: "Defines the contract for BLIP-2 (Bootstrapped Language-Image Pre-training 2) models."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Defines the contract for BLIP-2 (Bootstrapped Language-Image Pre-training 2) models.

## For Beginners

BLIP-2 is like having a smart translator between images and language!

Key innovation - the Q-Former:

- Uses special "query tokens" to ask questions about the image
- These queries learn to extract the most useful visual information
- The extracted features then connect to powerful language models (LLMs)

Why BLIP-2 is special:

- Uses frozen (pre-trained) image encoders like ViT-G
- Uses frozen LLMs like OPT or Flan-T5
- Only trains the small Q-Former bridge (much cheaper!)
- Gets state-of-the-art results with less compute

Use cases (same as BLIP but better):

- More accurate image captioning
- Better visual question answering
- More nuanced image-text understanding
- Can leverage larger LLMs for better generation

## How It Works

BLIP-2 is a more efficient and powerful successor to BLIP that uses a Q-Former
(Querying Transformer) to bridge frozen image encoders with frozen large language models.
This architecture enables better vision-language understanding with significantly
less training compute.

## Properties

| Property | Summary |
|:-----|:--------|
| `LanguageModelBackbone` | Gets the type of language model backbone used for generation. |
| `NumQueryTokens` | Gets the number of learnable query tokens used by the Q-Former. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AnswerQuestion(Tensor<>,String,Int32)` | Answers a question about an image using the LLM backend. |
| `ComputeContrastiveSimilarity(Tensor<>,String)` | Computes image-text contrastive similarity using Q-Former features. |
| `ComputeImageTextMatch(Tensor<>,String)` | Computes image-text matching score using the Q-Former's ITM head. |
| `ExtractQFormerFeatures(Tensor<>)` | Extracts visual features using the Q-Former's learnable queries. |
| `GenerateCaption(Tensor<>,String,Int32,Int32,Double)` | Generates a caption for an image using the LLM backend. |
| `GenerateCaptions(Tensor<>,Int32,String,Int32,Double,Double)` | Generates multiple diverse captions for an image. |
| `GenerateWithInstruction(Tensor<>,String,Int32)` | Generates text conditioned on both image and text context (instructed generation). |
| `GroundText(Tensor<>,String)` | Performs visual grounding to locate objects described in text. |
| `RetrieveImages(String,IEnumerable<Tensor<>>,Int32,Boolean,Int32)` | Retrieves the most relevant images for a text query. |
| `ZeroShotClassify(Tensor<>,IEnumerable<String>,Boolean)` | Performs zero-shot image classification using text prompts. |

