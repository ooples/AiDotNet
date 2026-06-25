---
title: "SentenceCompressor"
description: "Compressor that shortens sentences while preserving their core meaning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.PromptEngineering.Compression`

Compressor that shortens sentences while preserving their core meaning.

## For Beginners

Makes sentences shorter by keeping only the important parts.

Example:

What gets simplified:

- Introductory phrases removed
- Complex clauses simplified
- Passive voice converted to active (when possible)

## How It Works

This compressor analyzes sentence structure and removes non-essential clauses,
qualifiers, and modifiers while maintaining grammatical correctness and the
primary message.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SentenceCompressor(Func<String,Int32>)` | Initializes a new instance of the SentenceCompressor class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CompressCore(String,CompressionOptions)` | Compresses the prompt by simplifying sentences. |
| `ConsolidateWhitespace(String)` | Consolidates multiple whitespace characters. |
| `RemoveEmptyModifiers(String)` | Removes empty modifiers and intensifiers that don't add meaning. |
| `RemoveIntroductoryClauses(String)` | Removes common introductory clauses that don't add meaning. |
| `RemoveParentheticalClauses(String)` | Removes parenthetical and non-essential clauses. |
| `SimplifyPassiveVoice(String)` | Simplifies passive voice constructions to active voice where possible. |
| `SimplifyVerbPhrases(String)` | Simplifies verbose verb phrases. |

## Fields

| Field | Summary |
|:-----|:--------|
| `RegexTimeout` | Regex timeout to prevent ReDoS attacks. |

