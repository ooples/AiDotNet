---
title: "Word2VecVectorizer<T>"
description: "Converts text documents to dense vectors using Word2Vec-style word embeddings."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.TextVectorizers`

Converts text documents to dense vectors using Word2Vec-style word embeddings.

## For Beginners

Word2Vec captures word meaning in numbers:

- "king" - "man" + "woman" ≈ "queen" (famous example)
- Similar words have similar vectors
- Documents become the average of their word vectors
- Much smaller dimensions than bag-of-words (typically 100-300)

## How It Works

Word2Vec learns dense vector representations of words where semantically similar words
have similar vectors. This vectorizer trains word embeddings and represents documents
as the average (or weighted average) of their word vectors.

Two architectures are supported:

- CBOW (Continuous Bag of Words): Predicts target word from context
- Skip-gram: Predicts context words from target word

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `Word2VecVectorizer(Int32,Int32,Int32,Int32,Double,Int32,Word2VecArchitecture,Word2VecAggregation,Boolean,Nullable<Int32>,Func<String,IEnumerable<String>>,HashSet<String>,ITokenizer)` | Creates a new instance of `Word2VecVectorizer`. |

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
| `Fit(IEnumerable<String>)` | Trains Word2Vec embeddings on the corpus. |
| `GetWordVector(String)` | Gets the vector for a specific word. |
| `MostSimilar(String,Int32)` | Finds the most similar words to a given word. |
| `Transform(IEnumerable<String>)` | Transforms documents to dense vectors by aggregating word vectors. |

