---
title: "PIIEntity"
description: "Represents a detected PII entity in text."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Safety.Text`

Represents a detected PII entity in text.

## Properties

| Property | Summary |
|:-----|:--------|
| `Confidence` | Detection confidence between 0.0 and 1.0. |
| `EndIndex` | End character offset in the original text (exclusive, consistent with `SpanEnd`). |
| `StartIndex` | Start character offset in the original text. |
| `Type` | The type of PII detected (e.g., "Email", "SSN", "PhoneNumber"). |
| `Value` | The matched text value. |

