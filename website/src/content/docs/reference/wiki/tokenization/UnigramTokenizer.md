---
title: "UnigramTokenizer"
description: "Unigram Language Model tokenizer using probabilistic segmentation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Tokenization.Algorithms`

Unigram Language Model tokenizer using probabilistic segmentation.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `UnigramTokenizer(IVocabulary,Dictionary<String,Double>,SpecialTokens,Int32)` | Creates a new unigram tokenizer. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Tokenize(String)` | Tokenizes text using Viterbi algorithm for optimal segmentation. |
| `Train(IEnumerable<String>,Int32,SpecialTokens)` | Trains a unigram tokenizer from a corpus. |

