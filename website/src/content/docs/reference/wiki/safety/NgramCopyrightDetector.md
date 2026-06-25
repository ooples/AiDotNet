---
title: "NgramCopyrightDetector<T>"
description: "Detects potential copyright violations by measuring n-gram overlap with known copyrighted works."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Safety.Text`

Detects potential copyright violations by measuring n-gram overlap with known copyrighted works.

## For Beginners

If an AI reproduces large chunks of a book, article, or other
copyrighted work word-for-word, that's a copyright problem. This module checks how much
of the output matches known copyrighted texts by comparing sequences of words.

## How It Works

Computes word-level n-gram overlap between the model output and a corpus of known copyrighted
texts. High overlap (long verbatim sequences) indicates potential memorization or copyright
infringement. The detector uses sliding window analysis to identify the longest matching
subsequences.

**Detection approach:**

1. Extract word n-grams (n=4,5,6) from the output
2. Check against indexed n-grams from known copyrighted works
3. Compute overlap ratio — high overlap indicates potential infringement
4. Identify longest matching subsequence as evidence

**References:**

- DE-COP: Detecting copyrighted content via paraphrased permutations (2024, arxiv:2402.09910)
- Copyright pre-training data filtering (2025, arxiv:2512.02047)
- Machine unlearning to remove memorized copyrighted content (2024, arxiv:2412.18621)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `NgramCopyrightDetector(String[],String[],Double,Int32)` | Initializes a new n-gram copyright detector. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ModuleName` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `EvaluateText(String)` |  |

