---
title: "RedundancyRemovalCompressor"
description: "Compressor that removes redundant phrases and verbose language from prompts."
section: "API Reference"
---

`Models & Types` · `AiDotNet.PromptEngineering.Compression`

Compressor that removes redundant phrases and verbose language from prompts.

## For Beginners

Removes wordy phrases without changing meaning.

Example:

## How It Works

This compressor identifies and removes common patterns of redundant language
such as filler phrases, unnecessary qualifiers, and verbose constructions.
It preserves the semantic meaning while making the prompt more concise.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `RedundancyRemovalCompressor(Func<String,Int32>)` | Initializes a new instance of the RedundancyRemovalCompressor class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CompressCore(String,CompressionOptions)` | Compresses the prompt by removing redundant phrases. |
| `InitializeReplacements` | Initializes the replacement patterns for redundancy removal. |

## Fields

| Field | Summary |
|:-----|:--------|
| `RegexTimeout` | Regex timeout to prevent ReDoS attacks. |

