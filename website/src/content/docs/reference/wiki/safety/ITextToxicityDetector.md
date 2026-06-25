---
title: "ITextToxicityDetector<T>"
description: "Interface for toxicity detection modules that identify harmful, abusive, or toxic text content."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Safety.Text`

Interface for toxicity detection modules that identify harmful, abusive, or toxic text content.

## For Beginners

A toxicity detector checks if text contains harmful language like
insults, threats, or hate speech. Different implementations use different techniques —
some match known bad words (rule-based), others understand meaning (embedding-based),
and others use trained models (classifier-based). The ensemble combines all of these.

## How It Works

Toxicity detectors analyze text for hate speech, harassment, threats, profanity, and other
forms of harmful language. Implementations range from rule-based pattern matching to
embedding similarity and trained classifier approaches.

**References:**

- MetaTox knowledge graph for enhanced LLM toxicity detection (2024, arxiv:2412.15268)
- LLM-extracted rationales for interpretable hate speech detection (2024, arxiv:2403.12403)
- GPT-4o/LLaMA-3 zero-shot hate speech detection (2025, arxiv:2506.12744)

## Methods

| Method | Summary |
|:-----|:--------|
| `GetToxicityScore(String)` | Gets the toxicity score for the given text (0.0 = safe, 1.0 = maximally toxic). |

