---
title: "TokenizationConfig"
description: "Configuration options for tokenization in the prediction pipeline."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Tokenization.Configuration`

Configuration options for tokenization in the prediction pipeline.

## How It Works

**For Beginners:** Tokenization is the process of breaking text into smaller pieces (tokens)
that a machine learning model can understand. Think of it like breaking a sentence into words,
but sometimes words are further broken into subwords for better handling of unknown words.

## Properties

| Property | Summary |
|:-----|:--------|
| `AddSpecialTokens` | Gets or sets whether to automatically add special tokens (like [CLS], [SEP]) during encoding. |
| `DefaultEncodingOptions` | Gets or sets the default encoding options for tokenization. |
| `EnableCaching` | Gets or sets whether to cache tokenization results for repeated inputs. |
| `EnableParallelBatchProcessing` | Gets or sets whether to use parallel processing for batch tokenization. |
| `MaxLength` | Gets or sets the maximum sequence length for tokenization. |
| `Padding` | Gets or sets whether to pad sequences to the maximum length. |
| `PaddingSide` | Gets or sets the side on which to pad sequences ("left" or "right"). |
| `ParallelBatchThreshold` | Gets or sets the minimum batch size to trigger parallel processing. |
| `ReturnAttentionMask` | Gets or sets whether to return attention masks. |
| `ReturnTokenTypeIds` | Gets or sets whether to return token type IDs (for models like BERT with multiple segments). |
| `Truncation` | Gets or sets whether to truncate sequences that exceed max length. |
| `TruncationSide` | Gets or sets the side on which to truncate sequences ("left" or "right"). |

## Methods

| Method | Summary |
|:-----|:--------|
| `ForBert(Int32)` | Creates a configuration suitable for BERT-style models. |
| `ForCode(Int32)` | Creates a configuration suitable for code tokenization. |
| `ForGpt(Int32)` | Creates a configuration suitable for GPT-style models. |
| `ToEncodingOptions` | Creates encoding options based on this configuration. |

