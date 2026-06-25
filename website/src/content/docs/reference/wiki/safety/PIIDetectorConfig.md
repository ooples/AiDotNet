---
title: "PIIDetectorConfig"
description: "Configuration for PII detection modules."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Safety.Text`

Configuration for PII detection modules.

## For Beginners

Use this to configure which types of personal information
to detect (emails, phone numbers, SSNs, etc.) and what to do when PII is found
(mask it, hash it, remove it, etc.).

## Properties

| Property | Summary |
|:-----|:--------|
| `Categories` | PII categories to detect. |
| `CustomPatterns` | Custom regex patterns for additional PII types. |
| `Locale` | Locale for PII pattern matching (e.g., "en-US", "de-DE"). |
| `MinConfidence` | Minimum confidence to report a PII match. |
| `Redaction` | Redaction strategy to apply when PII is found. |

