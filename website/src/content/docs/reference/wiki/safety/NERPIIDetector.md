---
title: "NERPIIDetector<T>"
description: "Detects PII using Named Entity Recognition (NER) heuristics based on contextual patterns."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Safety.Text`

Detects PII using Named Entity Recognition (NER) heuristics based on contextual patterns.

## For Beginners

While regex can catch emails and phone numbers, it struggles with
person names (is "John Smith" a name or a product?). This module uses context clues — like
"Mr.", "CEO", "lives in" — to understand when words refer to real people, places, or
organizations.

## How It Works

Goes beyond simple regex by analyzing surrounding context to identify named entities that
represent PII. Uses pattern-based NER with capitalization analysis, contextual cues (titles,
prefixes), and statistical features to detect person names, organization names, and location
references that regex-based approaches miss.

**References:**

- PRvL: LLMs for contextual PII redaction outperform rule-based NER (2025, arxiv:2508.05545)
- CAPID: Context-aware PII detection reducing over-redaction in QA (2026, arxiv:2602.10074)
- Text anonymization survey bridging NER and LLMs (2025, arxiv:2508.21587)

## Properties

| Property | Summary |
|:-----|:--------|
| `ModuleName` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `EvaluateText(String)` |  |

