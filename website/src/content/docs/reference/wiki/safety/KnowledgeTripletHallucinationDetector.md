---
title: "KnowledgeTripletHallucinationDetector<T>"
description: "Detects hallucinations by extracting (subject, predicate, object) knowledge triplets and verifying them against reference documents."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Safety.Text`

Detects hallucinations by extracting (subject, predicate, object) knowledge triplets
and verifying them against reference documents.

## For Beginners

Imagine breaking every sentence into simple facts like
"X is related to Y in way Z". Then we check each fact against the source documents.
If a fact doesn't appear in any source, the AI probably made it up.

## How It Works

Parses the model output into knowledge triplets (e.g., "Paris – capital of – France")
and checks each triplet against the reference corpus. Triplets that cannot be grounded
in any reference document are flagged as potential hallucinations. This approach is more
precise than sentence-level grounding because it isolates individual factual claims.

**References:**

- RefChecker: Reference-based fine-grained hallucination via knowledge triplets (Amazon, 2024, arxiv:2405.14486)
- HHEM 2.1: Production-grade hallucination evaluation outperforming GPT-4 (Vectara, 2024)
- Triplet extraction for hallucination detection survey (2025, arxiv:2503.08100)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `KnowledgeTripletHallucinationDetector(String[],Double,Int32)` | Initializes a new knowledge triplet hallucination detector. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ModuleName` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `EvaluateText(String)` |  |
| `WithReferences(String[])` | Returns a new detector with updated reference documents. |

