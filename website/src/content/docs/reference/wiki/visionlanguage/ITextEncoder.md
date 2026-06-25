---
title: "ITextEncoder<T>"
description: "Interface for text encoders that extract feature representations from text."
section: "API Reference"
---

`Interfaces` · `AiDotNet.VisionLanguage.Interfaces`

Interface for text encoders that extract feature representations from text.

## For Beginners

A text encoder converts words and sentences into numbers that represent
their meaning. Similar sentences get similar numbers, which lets the model compare text with
images in the same mathematical space.

## How It Works

Text encoders transform text strings into compact feature representations (embeddings)
that capture semantic meaning. In vision-language models, text embeddings are aligned with
visual embeddings in a shared space for cross-modal tasks.

## Properties

| Property | Summary |
|:-----|:--------|
| `MaxSequenceLength` | Gets the maximum token sequence length supported by this encoder. |
| `TextEmbeddingDimension` | Gets the dimensionality of the output text embedding space. |

## Methods

| Method | Summary |
|:-----|:--------|
| `EncodeText(String)` | Encodes a text string into an embedding vector. |
| `EncodeTexts(String[])` | Encodes multiple text strings into embedding vectors. |

