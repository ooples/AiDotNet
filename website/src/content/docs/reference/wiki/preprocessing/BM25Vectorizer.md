---
title: "BM25Vectorizer<T>"
description: "Converts text documents to BM25-weighted feature vectors."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.TextVectorizers`

Converts text documents to BM25-weighted feature vectors.

## For Beginners

BM25 is like TF-IDF but smarter:

- Long documents don't unfairly dominate (length normalization)
- Repeating a word 100 times doesn't score 100x better than once (saturation)
- Used by Google, Elasticsearch, and most modern search engines
- Generally performs better than TF-IDF for search and retrieval tasks

## How It Works

BM25 (Best Matching 25) is an advanced ranking function used by search engines like
Elasticsearch and Lucene. It improves upon TF-IDF by adding document length normalization
and term frequency saturation.

The BM25 formula is:

Where:

- f(qi,D) = term frequency of qi in document D
- |D| = document length
- avgdl = average document length in the corpus
- k1 = term frequency saturation parameter (typically 1.2-2.0)
- b = document length normalization parameter (typically 0.75)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `BM25Vectorizer(Double,Double,Double,Int32,Double,Nullable<Int32>,Nullable<ValueTuple<Int32,Int32>>,Boolean,BM25Norm,Func<String,IEnumerable<String>>,HashSet<String>,ITokenizer)` | Creates a new instance of `BM25Vectorizer`. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AverageDocumentLength` | Gets the average document length in the corpus. |
| `IdfWeights` | Gets the IDF weights for each term. |
| `IsFitted` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Fit(IEnumerable<String>)` | Fits the vectorizer to the corpus. |
| `Transform(IEnumerable<String>)` | Transforms documents to BM25-weighted vectors. |

