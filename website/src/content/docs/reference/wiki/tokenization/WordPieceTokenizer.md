---
title: "WordPieceTokenizer"
description: "WordPiece tokenizer implementation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Tokenization.Algorithms`

WordPiece tokenizer implementation.
Used by BERT and similar models.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `WordPieceTokenizer(IVocabulary,SpecialTokens,String,Int32)` | Creates a new WordPiece tokenizer. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CleanupTokens(List<String>)` | Cleans up tokens and converts them back to text. |
| `Tokenize(String)` | Tokenizes text into WordPiece tokens. |
| `TokenizeWord(String)` | Tokenizes a single word using WordPiece algorithm. |
| `Train(IEnumerable<String>,Int32,SpecialTokens,String)` | Trains a WordPiece tokenizer from a corpus. |

