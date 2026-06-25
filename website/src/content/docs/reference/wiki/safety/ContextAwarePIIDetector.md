---
title: "ContextAwarePIIDetector<T>"
description: "Context-aware PII detector that reduces false positives by analyzing surrounding text."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Safety.Text`

Context-aware PII detector that reduces false positives by analyzing surrounding text.

## For Beginners

Sometimes a number that looks like a phone number is actually a
product code, and something that looks like an email is a filename. This module uses
the words around each detection to figure out if it's really personal information.

## How It Works

Wraps an inner PII detector and applies contextual validation to filter out false positives.
For example, "call me at 555-0100" is a phone number, but "HTTP error 404-0100" is not.
Analyzes a context window around each detection and uses heuristic rules to determine if
the detected pattern is actually PII in context.

**References:**

- CAPID: Context-aware PII detection reducing over-redaction in QA (2026, arxiv:2602.10074)
- HIPS method with GPT-4 for educational PII detection (2025, arxiv:2501.09765)
- False sense of privacy: surface-level PII removal insufficient (2025, arxiv:2504.21035)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ContextAwarePIIDetector(ITextSafetyModule<>,Int32)` | Initializes a new context-aware PII detector. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ModuleName` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `EvaluateText(String)` |  |

