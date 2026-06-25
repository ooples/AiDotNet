---
title: "ClipTokenizerFactory"
description: "Factory for creating CLIP-compatible tokenizers."
section: "API Reference"
---

`Helpers & Utilities` · `AiDotNet.Tokenization`

Factory for creating CLIP-compatible tokenizers.

## For Beginners

CLIP needs a special tokenizer to break text into pieces.

A tokenizer factory is like a tool shop that builds tokenizers:

1. You can load a pretrained tokenizer (recommended for production)
2. You can create a simple tokenizer for testing
3. The factory handles all the configuration details

Example usage:

## How It Works

CLIP uses a BPE tokenizer with a vocabulary of 49408 tokens. This factory
provides methods to create tokenizers from pretrained vocabulary files
or to use a default configuration for testing.

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateSimple(IEnumerable<String>,Int32)` | Creates a simple CLIP tokenizer for testing without pretrained files. |
| `FromPretrained(String,String)` | Creates a CLIP tokenizer from pretrained vocabulary and merge files. |
| `GetDefaultEncodingOptions(Int32)` | Gets the default encoding options for CLIP text encoding. |
| `IsClipCompatible(ITokenizer)` | Validates that a tokenizer is compatible with CLIP. |

## Fields

| Field | Summary |
|:-----|:--------|
| `ClipPattern` | The CLIP-specific pre-tokenization pattern. |
| `DefaultMaxLength` | The default maximum sequence length for CLIP text encoder. |
| `DefaultVocabSize` | The default vocabulary size for CLIP models. |

