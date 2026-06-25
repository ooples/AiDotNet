---
title: "RegexPIIDetector<T>"
description: "Regex-based PII (Personally Identifiable Information) detector for common PII patterns."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Safety.Text`

Regex-based PII (Personally Identifiable Information) detector for common PII patterns.

## For Beginners

This detector scans text for personal information that shouldn't
be shared — like email addresses, phone numbers, credit card numbers, and Social Security
Numbers. When found, it reports the location so the information can be redacted.

## How It Works

Detects personally identifiable information using curated regex patterns for common PII
types: email addresses, phone numbers, Social Security Numbers, credit card numbers,
IP addresses, API keys, and more.

**Limitations:** Regex-based detection has high precision for structured PII (emails,
SSNs, credit cards) but cannot detect unstructured PII (names, addresses in free text).
For comprehensive PII detection, combine with a NER-based detector.

**References:**

- PRvL: LLMs for contextual PII redaction outperform rule-based NER (2025, arxiv:2508.05545)
- CAPID: Context-aware PII detection reducing over-redaction (2026, arxiv:2602.10074)
- Hybrid multilingual PII detection evaluation (2025, arxiv:2510.07551)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `RegexPIIDetector` | Initializes a new instance of the regex-based PII detector. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ModuleName` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `EvaluateText(String)` |  |
| `MaskPII(String)` | Masks PII for safe inclusion in findings (shows first/last char only). |

