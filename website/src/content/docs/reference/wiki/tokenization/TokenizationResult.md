---
title: "TokenizationResult"
description: "Represents the result of tokenizing text, including token IDs, tokens, and attention masks."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Tokenization.Models`

Represents the result of tokenizing text, including token IDs, tokens, and attention masks.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TokenizationResult` | Creates an empty tokenization result. |
| `TokenizationResult(List<String>,List<Int32>)` | Creates a tokenization result with the specified tokens and IDs. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AttentionMask` | Gets or sets the attention mask (1 for real tokens, 0 for padding). |
| `Length` | Gets the number of tokens (excluding padding). |
| `Metadata` | Gets or sets additional metadata. |
| `Offsets` | Gets or sets character-level offsets for each token. |
| `PositionIds` | Gets or sets the position IDs for positional embeddings. |
| `TokenIds` | Gets or sets the token IDs. |
| `TokenTypeIds` | Gets or sets the token type IDs (for models that support multiple segments). |
| `Tokens` | Gets or sets the actual tokens (subword strings). |
| `TotalLength` | Gets the total number of token IDs (including padding). |

