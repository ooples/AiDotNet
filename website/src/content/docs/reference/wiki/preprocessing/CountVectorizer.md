---
title: "CountVectorizer<T>"
description: "Converts a collection of text documents to a matrix of token counts."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.TextVectorizers`

Converts a collection of text documents to a matrix of token counts.

## For Beginners

CountVectorizer turns text into numbers:

- Each unique word becomes a column
- Each document becomes a row
- Values are word counts

Example: ["I like cats", "I like dogs"] becomes:
| I | like | cats | dogs |
Doc 1: | 1 | 1 | 1 | 0 |
Doc 2: | 1 | 1 | 0 | 1 |

## How It Works

CountVectorizer tokenizes text documents and builds a vocabulary, then
encodes each document as a count vector (bag of words representation).

Features include:

- Customizable tokenization
- N-gram support (unigrams, bigrams, etc.)
- Minimum and maximum document frequency thresholds
- Maximum vocabulary size

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CountVectorizer(Int32,Double,Nullable<Int32>,Nullable<ValueTuple<Int32,Int32>>,Boolean,Boolean,Func<String,IEnumerable<String>>,HashSet<String>,ITokenizer)` | Creates a new instance of `CountVectorizer`. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Fit(IEnumerable<String>)` | Fits the vectorizer to the corpus. |
| `Transform(IEnumerable<String>)` | Transforms documents to count vectors. |

