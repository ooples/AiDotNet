---
title: "EmbeddingToxicityDetector<T>"
description: "Detects toxic text using embedding-based cosine similarity to known toxic concept vectors."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Safety.Text`

Detects toxic text using embedding-based cosine similarity to known toxic concept vectors.

## For Beginners

Instead of looking for specific bad words, this module converts text
into a mathematical representation (embedding) and measures how "close" it is to known
examples of toxic content. This catches rephrasings, misspellings, and subtle toxicity
that keyword matching would miss.

## How It Works

Computes a lightweight embedding (character n-gram hash vectors) of the input text and
measures cosine similarity against pre-built concept vectors for known toxic categories.
This approach catches semantically similar content that regex-based approaches miss.

**References:**

- MetaTox knowledge graph for enhanced LLM toxicity detection (2024, arxiv:2412.15268)
- LLM-extracted rationales for interpretable hate speech detection (2024, arxiv:2403.12403)
- GPT-4o/LLaMA-3 zero-shot hate speech detection (2025, arxiv:2506.12744)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `EmbeddingToxicityDetector(Double,Int32)` | Initializes a new embedding-based toxicity detector. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ModuleName` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeConceptEmbedding(String[])` | Computes a concept embedding by averaging embeddings of representative phrases. |
| `ComputeTextEmbedding(String)` | Computes a character n-gram hash embedding for the given text. |
| `EvaluateText(String)` |  |

