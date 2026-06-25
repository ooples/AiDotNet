---
title: "IConditioningModule<T>"
description: "Interface for conditioning modules that encode various inputs into embeddings for diffusion models."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Interface for conditioning modules that encode various inputs into embeddings for diffusion models.

## For Beginners

A conditioning module is like a "translator" that converts your input
(like a text prompt) into a format the diffusion model can understand.

Common types of conditioning:

1. Text conditioning (CLIP, T5): "A cat sitting on a couch" → embedding vectors
2. Image conditioning (IP-Adapter): An image → embedding vectors for style/content
3. Control conditioning (ControlNet): Depth maps, edges, poses → spatial guidance

Why conditioning matters:

- Without conditioning: Model generates random images
- With text conditioning: Model generates images matching your description
- With image conditioning: Model preserves style or content from reference images
- With control conditioning: Model follows spatial structure (poses, edges, depth)

Different conditioning methods:

- Cross-attention: Text embeddings attend to image features (most common for text)
- Addition/Concatenation: Add or concat embeddings to time embedding
- Spatial: Add control signals directly to features at each resolution

## How It Works

Conditioning modules convert various types of input (text, images, audio, etc.) into
embedding tensors that guide the diffusion process. They are essential for controlled
generation like text-to-image, image-to-image, or style transfer.

## Properties

| Property | Summary |
|:-----|:--------|
| `ConditioningType` | Gets the type of conditioning this module provides. |
| `EmbeddingDimension` | Gets the dimension of the output embeddings. |
| `MaxSequenceLength` | Gets the maximum sequence length for text input. |
| `ProducesPooledOutput` | Gets whether this module produces pooled (global) or sequence embeddings. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Encode(Tensor<>)` | Encodes the input into conditioning embeddings. |
| `EncodeText(Tensor<>,Tensor<>)` | Encodes text input (convenience method for text conditioning). |
| `GetPooledEmbedding(Tensor<>)` | Gets the pooled (global) embedding from sequence embeddings. |
| `GetUnconditionalEmbedding(Int32)` | Gets unconditioned (null) embeddings for classifier-free guidance. |
| `Tokenize(String)` | Tokenizes text input (for text conditioning modules). |
| `TokenizeBatch(String[])` | Tokenizes a batch of text inputs. |

