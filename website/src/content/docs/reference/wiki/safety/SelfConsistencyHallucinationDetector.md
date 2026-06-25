---
title: "SelfConsistencyHallucinationDetector<T>"
description: "Detects hallucinations by checking internal consistency of claims within the text."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Safety.Text`

Detects hallucinations by checking internal consistency of claims within the text.

## For Beginners

If an AI says "the building was built in 1990" and later says
"the building was constructed in 2005", those statements contradict each other. This module
finds such contradictions, which often indicate the AI is making things up.

## How It Works

Analyzes model output for internal contradictions and inconsistencies. When a model
hallucinates, it often produces statements that contradict each other or contain
logically impossible combinations. This detector identifies such patterns by comparing
sentence-level embeddings within the same document.

**Detection approach:**

1. Split text into sentences (claims)
2. Compute embeddings for each sentence
3. Compare all pairs for semantic contradiction signals
4. Flag texts with high contradiction rates

**References:**

- SelfCheckGPT: Zero-resource hallucination detection (2023)
- ReDeEP: Hallucination detection in RAG systems (ICLR 2025)
- Hallucination survey: faithfulness vs factuality taxonomy (2025, arxiv:2510.06265)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SelfConsistencyHallucinationDetector(Double,Int32)` | Initializes a new self-consistency hallucination detector. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ModuleName` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `EvaluateText(String)` |  |

