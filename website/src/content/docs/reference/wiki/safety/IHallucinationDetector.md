---
title: "IHallucinationDetector<T>"
description: "Interface for hallucination detection modules that identify fabricated or unfaithful content."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Safety.Text`

Interface for hallucination detection modules that identify fabricated or unfaithful content.

## For Beginners

A hallucination detector checks if an AI made something up.
It can compare the AI's output against source documents, check if the AI contradicts
itself, or verify specific facts. This helps ensure AI outputs are trustworthy.

## How It Works

Hallucination detectors analyze model outputs to identify claims that are not grounded
in source material or are factually inconsistent. Approaches include reference-based
comparison, self-consistency checking, knowledge triplet extraction, and NLI entailment.

**References:**

- RefChecker: Knowledge triplet-based detection (Amazon, 2024, arxiv:2405.14486)
- HHEM 2.1/2.3: Production-grade detection (Vectara, 2024-2025)
- ReDeEP: Hallucination detection in RAG systems (ICLR 2025)

## Methods

| Method | Summary |
|:-----|:--------|
| `EvaluateAgainstReference(String,String)` | Evaluates text against reference content for faithfulness. |
| `GetHallucinationScore(String)` | Gets the hallucination likelihood score (0.0 = grounded, 1.0 = fabricated). |

