---
title: "ReferenceBasedHallucinationDetector<T>"
description: "Detects hallucinations by comparing model output against provided reference documents."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Safety.Text`

Detects hallucinations by comparing model output against provided reference documents.

## For Beginners

When an AI makes a claim, this module checks whether that claim
is supported by the source documents. If the AI says something that isn't in the sources,
it's likely a "hallucination" — something the AI made up.

## How It Works

Measures the overlap between claims in the model output and information present in reference
documents. Claims that cannot be grounded in the reference material are flagged as potential
hallucinations. Uses n-gram overlap and embedding similarity for grounding verification.

**References:**

- RefChecker: Knowledge triplet-based detection (Amazon, 2024, arxiv:2405.14486)
- HHEM 2.1/2.3: Production-grade detection beating GPT-4 (Vectara, 2024-2025)
- FaithBench: Benchmarking hallucination in summarization (2025, arxiv:2505.04847)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ReferenceBasedHallucinationDetector(String[],Double,Int32)` | Initializes a new reference-based hallucination detector. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ModuleName` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `EvaluateText(String)` |  |
| `WithReferences(String[])` | Sets or updates the reference documents for grounding checks. |

