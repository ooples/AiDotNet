---
title: "BpeTokenizer"
description: "Byte-Pair Encoding (BPE) tokenizer implementation for subword tokenization."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Tokenization.Algorithms`

Byte-Pair Encoding (BPE) tokenizer implementation for subword tokenization.

## For Beginners

BPE is like learning common letter combinations. Imagine
you're creating shorthand notes:

1. Start with individual letters: "t", "h", "e", " ", "c", "a", "t"
2. Notice "th" appears often, so create a symbol for it: "th", "e", " ", ...
3. Notice "the" appears often, merge again: "the", " ", "cat"
4. Keep merging until you have a good vocabulary size

This way, common words like "the" become single tokens, while rare words like
"cryptocurrency" might be split into "crypt" + "ocurrency" or similar subwords.

Benefits:

- No out-of-vocabulary words (any text can be tokenized)
- Common words are single tokens (efficient)
- Rare words are split into meaningful subwords (handles new words)

Example tokenization of "unhappiness":

- Full word not in vocabulary, so split into subwords
- Possible result: ["un", "happiness"] or ["un", "happy", "ness"]

## How It Works

BPE is a data compression algorithm adapted for NLP that learns to merge frequent
character sequences into subword units. It's used by GPT, GPT-2, GPT-3, and many
other modern language models.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `BpeTokenizer(IVocabulary,Dictionary<ValueTuple<String,String>,Int32>,SpecialTokens,String)` | Creates a new BPE tokenizer with the specified vocabulary and merge rules. |

## Methods

| Method | Summary |
|:-----|:--------|
| `BpeEncode(String)` | Applies BPE encoding to a word. |
| `CleanupTokens(List<String>)` | Cleans up tokens and converts them back to text. |
| `Tokenize(String)` | Tokenizes text into BPE tokens. |
| `Train(IEnumerable<String>,Int32,SpecialTokens,String)` | Trains a BPE tokenizer from a text corpus by learning merge rules. |

