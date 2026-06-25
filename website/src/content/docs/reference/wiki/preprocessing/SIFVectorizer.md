---
title: "SIFVectorizer<T>"
description: "Converts text documents to sentence embeddings using Smooth Inverse Frequency (SIF)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.TextVectorizers`

Converts text documents to sentence embeddings using Smooth Inverse Frequency (SIF).

## For Beginners

SIF creates sentence embeddings that are surprisingly good:

- Simple: just weighted averaging of word vectors
- Effective: often competitive with complex deep learning methods
- Fast: no neural network inference required
- Requires pre-trained word vectors (Word2Vec, GloVe, etc.)

## How It Works

SIF is a simple but effective method for creating sentence embeddings from word vectors.
It computes a weighted average of word vectors using smooth inverse frequency weights,
then removes the first principal component to create better sentence representations.

The algorithm:

1. Compute weighted average: v_s = (1/|s|) * Σ (a / (a + p(w))) * v_w
2. Remove first principal component: v_s = v_s - u * u^T * v_s

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SIFVectorizer(Dictionary<String,Double[]>,Double,Boolean,Int32,Boolean,Func<String,IEnumerable<String>>,HashSet<String>,ITokenizer)` | Creates a new instance of `SIFVectorizer`. |

## Properties

| Property | Summary |
|:-----|:--------|
| `FeatureCount` |  |
| `IsFitted` |  |
| `VectorSize` | Gets the vector dimensionality. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Fit(IEnumerable<String>)` | Fits the SIF vectorizer by computing word frequencies and principal components. |
| `Transform(IEnumerable<String>)` | Transforms documents to SIF sentence embeddings. |

