---
title: "CharNGramVectorizer<T>"
description: "Converts text documents to character-level n-gram feature vectors."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.TextVectorizers`

Converts text documents to character-level n-gram feature vectors.

## For Beginners

Character n-grams capture patterns at the letter level:

- "hello" with (2,3) n-grams produces: "he", "el", "ll", "lo", "hel", "ell", "llo"
- Robust to spelling mistakes ("color" and "colour" share many character n-grams)
- Works well for author identification and language detection
- Can capture morphological patterns (prefixes, suffixes)

## How It Works

CharNGramVectorizer creates features from character sequences rather than words.
This is particularly useful for:

- Handling misspellings and typos
- Languages without clear word boundaries
- Capturing subword patterns
- Short text classification (tweets, SMS)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CharNGramVectorizer(Nullable<ValueTuple<Int32,Int32>>,Boolean,Int32,Double,Nullable<Int32>,Boolean,CharNGramNorm,HashSet<String>,ITokenizer)` | Creates a new instance of `CharNGramVectorizer`. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Fit(IEnumerable<String>)` | Fits the vectorizer to the corpus. |
| `GenerateCharNGrams(String)` | Generates character n-grams from text. |
| `Transform(IEnumerable<String>)` | Transforms documents to character n-gram vectors. |

