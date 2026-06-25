---
title: "PhonemeTokenizer"
description: "Phoneme-based tokenizer for speech synthesis (TTS) applications."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Tokenization.Specialized`

Phoneme-based tokenizer for speech synthesis (TTS) applications.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PhonemeTokenizer(IVocabulary,Dictionary<String,String>,SpecialTokens,PhonemeSet)` | Creates a new phoneme tokenizer. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateARPAbet(SpecialTokens)` | Creates a phoneme tokenizer with ARPAbet phonemes. |
| `Tokenize(String)` | Tokenizes text into phonemes. |

