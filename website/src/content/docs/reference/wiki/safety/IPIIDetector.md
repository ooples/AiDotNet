---
title: "IPIIDetector<T>"
description: "Interface for PII (Personally Identifiable Information) detection modules."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Safety.Text`

Interface for PII (Personally Identifiable Information) detection modules.

## For Beginners

A PII detector finds personal information like email addresses,
phone numbers, and social security numbers in text. This helps prevent accidentally
sharing private data through AI outputs.

## How It Works

PII detectors identify sensitive personal information in text including names, emails,
phone numbers, SSNs, credit cards, addresses, API keys, and other data that could
compromise privacy if exposed.

**References:**

- PRvL: LLMs for contextual PII redaction outperform rule-based NER (2025, arxiv:2508.05545)
- CAPID: Context-aware PII detection reducing over-redaction (2026, arxiv:2602.10074)
- Hybrid multilingual PII detection (2025, arxiv:2510.07551)

## Methods

| Method | Summary |
|:-----|:--------|
| `DetectPII(String)` | Detects PII entities in the given text and returns their locations and types. |

