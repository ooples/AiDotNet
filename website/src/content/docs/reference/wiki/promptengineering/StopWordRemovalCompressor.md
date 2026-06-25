---
title: "StopWordRemovalCompressor"
description: "Compressor that removes common stop words to reduce prompt length."
section: "API Reference"
---

`Models & Types` · `AiDotNet.PromptEngineering.Compression`

Compressor that removes common stop words to reduce prompt length.

## For Beginners

Removes common words like "the", "a", "is" to make prompts shorter.

Example:

When to use:

- When maximum compression is needed
- When the AI can infer meaning from keywords
- Not recommended for conversational prompts

## How It Works

This compressor removes frequently occurring words that often don't add significant
meaning to the prompt (articles, prepositions, auxiliary verbs, etc.). It's more aggressive
than redundancy removal and may slightly reduce readability.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `StopWordRemovalCompressor(AggressivenessLevel,Func<String,Int32>)` | Initializes a new instance of the StopWordRemovalCompressor class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CompressCore(String,CompressionOptions)` | Compresses the prompt by removing stop words. |
| `InitializeStopWords(AggressivenessLevel)` | Initializes the set of stop words based on aggressiveness level. |

## Fields

| Field | Summary |
|:-----|:--------|
| `RegexTimeout` | Regex timeout to prevent ReDoS attacks. |

