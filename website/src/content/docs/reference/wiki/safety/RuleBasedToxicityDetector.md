---
title: "RuleBasedToxicityDetector<T>"
description: "Rule-based toxicity detector using pattern matching for harmful content detection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Safety.Text`

Rule-based toxicity detector using pattern matching for harmful content detection.

## For Beginners

This is the simplest toxicity detector — it uses pattern matching
(like a word filter) to catch harmful content. It's fast and reliable for obvious cases
but may miss subtle or context-dependent toxicity. For production use with higher accuracy,
combine with an `EmbeddingToxicityDetector` or `ClassifierToxicityDetector`.

## How It Works

This detector uses curated regex patterns to identify toxic, hateful, violent, and
other harmful text content. It provides fast, deterministic detection without requiring
any ML model or external dependencies.

**Design Decisions:**

- Regex timeout of 100ms prevents ReDoS attacks
- Patterns use word boundaries (\b) to reduce false positives
- Each pattern is associated with a specific SafetyCategory for granular reporting
- Compiled regex for performance on repeated evaluations

**References:**

- GPT-3.5/Llama 2 achieving 80-90% accuracy in hate speech identification

(2024, arxiv:2403.08035) — rule-based provides baseline; ML provides higher accuracy

- MetaTox knowledge graph for enhanced LLM toxicity detection

(2024, arxiv:2412.15268) — knowledge-graph augmented approach for future enhancement

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `RuleBasedToxicityDetector(Double)` | Initializes a new instance of the rule-based toxicity detector. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ModuleName` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `EvaluateText(String)` |  |

