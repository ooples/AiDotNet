---
title: "CompositePIIDetector<T>"
description: "Combines multiple PII detection strategies into a unified detector with deduplication."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Safety.Text`

Combines multiple PII detection strategies into a unified detector with deduplication.

## For Beginners

Different PII detection methods are good at finding different types
of personal information. This module runs all of them and combines the results, removing
duplicates so you get one clean list of all PII found.

## How It Works

Runs multiple PII detectors (regex, NER, context-aware) and merges their results with
span-level deduplication. When multiple detectors flag the same text region, the finding
with the highest confidence is kept. This provides comprehensive coverage while avoiding
duplicate alerts.

**References:**

- Hybrid multilingual PII detection (2025, arxiv:2510.07551)
- PRvL: LLMs for contextual PII redaction outperform rule-based NER (2025, arxiv:2508.05545)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CompositePIIDetector` | Initializes a new composite PII detector with default sub-detectors. |
| `CompositePIIDetector(ITextSafetyModule<>[])` | Initializes a new composite PII detector with custom sub-detectors. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ModuleName` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `EvaluateText(String)` |  |

