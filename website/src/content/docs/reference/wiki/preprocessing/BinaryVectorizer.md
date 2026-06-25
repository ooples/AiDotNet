---
title: "BinaryVectorizer<T>"
description: "Converts text documents to binary feature vectors (presence/absence encoding)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.TextVectorizers`

Converts text documents to binary feature vectors (presence/absence encoding).

## For Beginners

Binary encoding is the simplest text representation:

- Each word is either present (1) or absent (0) in a document
- Word frequency is ignored (appearing 10 times = appearing once)
- Fast and memory-efficient
- Works well when word presence matters more than frequency
- Common in document classification and spam detection

## How It Works

BinaryVectorizer creates a simple presence/absence encoding where each feature
is 1 if the term appears in the document and 0 otherwise. Unlike CountVectorizer
with binary=true, this vectorizer is optimized specifically for binary encoding.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `BinaryVectorizer(Int32,Double,Nullable<Int32>,Nullable<ValueTuple<Int32,Int32>>,Boolean,Func<String,IEnumerable<String>>,HashSet<String>,ITokenizer)` | Creates a new instance of `BinaryVectorizer`. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Fit(IEnumerable<String>)` | Fits the vectorizer to the corpus. |
| `Transform(IEnumerable<String>)` | Transforms documents to binary vectors. |

