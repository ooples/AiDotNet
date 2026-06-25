---
title: "TokenizerBase"
description: "Base class for tokenizers providing common functionality."
section: "API Reference"
---

`Base Classes` · `AiDotNet.Tokenization.Core`

Base class for tokenizers providing common functionality.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TokenizerBase(IVocabulary,SpecialTokens)` | Initializes a new instance of the TokenizerBase class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `SpecialTokens` | Gets the special tokens. |
| `Vocabulary` | Gets the vocabulary. |
| `VocabularySize` | Gets the vocabulary size. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AddSpecialTokensToSequence(List<String>)` | Adds special tokens to a sequence. |
| `CleanupTokens(List<String>)` | Cleans up tokens and converts them back to text (must be implemented by derived classes). |
| `ConvertIdsToTokens(List<Int32>)` | Converts token IDs to tokens. |
| `ConvertTokensToIds(List<String>)` | Converts tokens to token IDs. |
| `Decode(List<Int32>,Boolean)` | Decodes token IDs back into text. |
| `DecodeBatch(List<List<Int32>>,Boolean)` | Decodes multiple sequences of token IDs back into text. |
| `Encode(String,EncodingOptions)` | Encodes text into tokens. |
| `EncodeBatch(List<String>,EncodingOptions)` | Encodes multiple texts into tokens. |
| `Tokenize(String)` | Tokenizes text into subword tokens (must be implemented by derived classes). |
| `TruncateSequence(List<String>,Int32,String)` | Truncates a sequence to a maximum length. |

