---
title: "LemmatizationQueryProcessor"
description: "Reduces words to their base or dictionary form (lemma) for better matching."
section: "API Reference"
---

`Models & Types` · `AiDotNet.RetrievalAugmentedGeneration.QueryProcessors`

Reduces words to their base or dictionary form (lemma) for better matching.

## For Beginners

Converts words to their basic form.

Examples:

- "running" → "run"
- "better" → "good"
- "was" → "be"
- "cars" → "car"

This helps match documents that use different forms of the same word!

## How It Works

Lemmatization transforms words to their base form considering the word's meaning.
Unlike stemming, it produces valid dictionary words.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LemmatizationQueryProcessor(Dictionary<String,String>)` | Initializes a new instance of the LemmatizationQueryProcessor class. |

## Fields

| Field | Summary |
|:-----|:--------|
| `RegexTimeout` | Timeout for regex operations to prevent ReDoS attacks. |

