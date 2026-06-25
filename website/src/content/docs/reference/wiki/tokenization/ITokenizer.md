---
title: "ITokenizer"
description: "Interface for text tokenizers."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Tokenization.Interfaces`

Interface for text tokenizers.

## Properties

| Property | Summary |
|:-----|:--------|
| `SpecialTokens` | Gets the special tokens. |
| `Vocabulary` | Gets the vocabulary. |
| `VocabularySize` | Gets the vocabulary size. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ConvertIdsToTokens(List<Int32>)` | Converts token IDs to tokens. |
| `ConvertTokensToIds(List<String>)` | Converts tokens to token IDs. |
| `Decode(List<Int32>,Boolean)` | Decodes token IDs back into text. |
| `DecodeBatch(List<List<Int32>>,Boolean)` | Decodes multiple sequences of token IDs back into text. |
| `Encode(String,EncodingOptions)` | Encodes text into tokens. |
| `EncodeBatch(List<String>,EncodingOptions)` | Encodes multiple texts into tokens. |
| `Tokenize(String)` | Tokenizes text into subword tokens (without converting to IDs). |

