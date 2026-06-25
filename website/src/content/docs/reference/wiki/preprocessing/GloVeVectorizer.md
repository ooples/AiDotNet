---
title: "GloVeVectorizer<T>"
description: "Converts text documents to dense vectors using GloVe-style word embeddings."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.TextVectorizers`

Converts text documents to dense vectors using GloVe-style word embeddings.

## For Beginners

GloVe combines the best of both worlds:

- Uses global word co-occurrence statistics (like LSA)
- Produces dense vectors (like Word2Vec)
- Often produces high-quality embeddings with less training time
- The resulting word relationships are mathematically meaningful

## How It Works

GloVe (Global Vectors for Word Representation) learns word vectors by factorizing
the word-word co-occurrence matrix. Unlike Word2Vec which uses local context windows,
GloVe leverages global corpus statistics for potentially better embeddings.

The objective function is:

Where Xij is the co-occurrence count and f is a weighting function.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GloVeVectorizer(Int32,Int32,Int32,Int32,Double,Double,Double,Word2VecAggregation,Boolean,Nullable<Int32>,Func<String,IEnumerable<String>>,HashSet<String>,ITokenizer)` | Creates a new instance of `GloVeVectorizer`. |

## Properties

| Property | Summary |
|:-----|:--------|
| `FeatureCount` |  |
| `IsFitted` |  |
| `VectorSize` | Gets the vector dimensionality. |
| `WordVectors` | Gets the learned word vectors. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Fit(IEnumerable<String>)` | Trains GloVe embeddings on the corpus. |
| `GetWordVector(String)` | Gets the vector for a specific word. |
| `MostSimilar(String,Int32)` | Finds the most similar words to a given word. |
| `Transform(IEnumerable<String>)` | Transforms documents to dense vectors by aggregating word vectors. |

