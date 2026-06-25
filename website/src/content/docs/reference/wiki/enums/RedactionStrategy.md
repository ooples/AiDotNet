---
title: "RedactionStrategy"
description: "Strategy for redacting detected PII from text."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums`

Strategy for redacting detected PII from text.

## For Beginners

When personal information is found in text, you can choose
how to handle it. Masking replaces it with asterisks, hashing replaces it with a
consistent hash, replacement uses a placeholder, and removal deletes it entirely.

## Fields

| Field | Summary |
|:-----|:--------|
| `Hash` | Replace PII with a consistent hash (e.g., "John" → "[HASH:a1b2c3]"). |
| `Mask` | Replace PII with asterisks (e.g., "John" → "****"). |
| `Remove` | Remove PII entirely from the text. |
| `Replace` | Replace PII with a type placeholder (e.g., "John" → "[PERSON]"). |
| `Tokenize` | Replace PII with a reversible token for later de-anonymization. |

