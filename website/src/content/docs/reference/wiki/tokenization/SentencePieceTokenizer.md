---
title: "SentencePieceTokenizer"
description: "SentencePiece tokenizer implementation using Unigram language model."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Tokenization.Algorithms`

SentencePiece tokenizer implementation using Unigram language model.
Used for multilingual models and language-agnostic tokenization.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SentencePieceTokenizer(IVocabulary,Dictionary<String,Double>,SpecialTokens,Boolean)` | Creates a new SentencePiece tokenizer. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CleanupTokens(List<String>)` | Cleans up tokens and converts them back to text. |
| `Tokenize(String)` | Tokenizes text into SentencePiece tokens. |
| `Train(IEnumerable<String>,Int32,SpecialTokens,Double)` | Trains a SentencePiece tokenizer using Unigram language model. |
| `ViterbiSegmentation(String)` | Performs Viterbi segmentation to find the best tokenization. |

