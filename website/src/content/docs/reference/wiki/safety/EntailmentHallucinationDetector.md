---
title: "EntailmentHallucinationDetector<T>"
description: "Detects hallucinations using textual entailment (NLI) principles: checking whether reference documents entail (support) each claim in the model output."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Safety.Text`

Detects hallucinations using textual entailment (NLI) principles: checking whether
reference documents entail (support) each claim in the model output.

## For Beginners

Given source documents and an AI response, this module checks whether
each statement in the response logically follows from the sources. Statements that contradict
or go beyond the sources are flagged as hallucinations.

## How It Works

For each sentence in the model output, computes an entailment score against each reference
document. A sentence is "entailed" if it is logically supported by the reference. Sentences
that are contradicted or neutral (not supported) are flagged as potential hallucinations.
The approach uses lexical overlap, negation detection, and entity alignment as lightweight
proxies for full NLI model inference.

**References:**

- TRUE: Re-evaluating factual consistency via NLI (Google, 2023)
- SummaC: NLI-based consistency benchmark for summarization (2022)
- MiniCheck: Efficient NLI fact-checking grounding (2024, arxiv:2404.10774)
- AlignScore: Unified alignment for factual consistency (ACL 2023)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `EntailmentHallucinationDetector(String[],Double,Int32)` | Initializes a new entailment-based hallucination detector. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ModuleName` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `EvaluateText(String)` |  |
| `WithReferences(String[])` | Returns a new detector with updated reference documents. |

