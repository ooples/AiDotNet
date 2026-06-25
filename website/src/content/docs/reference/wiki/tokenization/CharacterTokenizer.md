---
title: "CharacterTokenizer"
description: "Character-level tokenizer that splits text into individual characters."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Tokenization.Algorithms`

Character-level tokenizer that splits text into individual characters.
Useful for character-based language models and some RNN architectures.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CharacterTokenizer(IVocabulary,SpecialTokens,Boolean,Boolean)` | Creates a new character tokenizer. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CleanupTokens(List<String>)` | Cleans up tokens and converts them back to text. |
| `CreateAscii(SpecialTokens,Boolean)` | Creates a character tokenizer with ASCII printable characters. |
| `Tokenize(String)` | Tokenizes text into individual characters. |
| `Train(IEnumerable<String>,SpecialTokens,Boolean,Int32)` | Trains a character tokenizer from a corpus. |

