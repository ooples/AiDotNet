---
title: "EmbeddingCopyrightDetector<T>"
description: "Detects potential copyright violations using embedding-based semantic similarity to known copyrighted works."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Safety.Text`

Detects potential copyright violations using embedding-based semantic similarity
to known copyrighted works.

## For Beginners

If someone rewrites a copyrighted book using different words but the
same ideas and structure, n-gram matching won't catch it. This module converts text into
mathematical representations that capture meaning, so it can detect "same content, different
words" situations.

## How It Works

While `NgramCopyrightDetector` catches verbatim copying, this detector catches
paraphrased or semantically similar reproductions of copyrighted content. Each reference work
is split into passages, embedded into a fixed-dimensional vector space, and compared against
the model output using cosine similarity. High similarity to specific passages indicates
potential memorization even when exact wording differs.

**References:**

- DE-COP: Detecting copyrighted content via paraphrased permutations (2024, arxiv:2402.09910)
- CopyBench: Measuring literal and non-literal copyright memorization (2024, arxiv:2407.07087)
- BookMIA: Practical membership inference for book-level memorization (2024, arxiv:2401.15588)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `EmbeddingCopyrightDetector(String[],String[],Double,Int32,Int32)` | Initializes a new embedding-based copyright detector. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ModuleName` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `EvaluateText(String)` |  |

