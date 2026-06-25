---
title: "PretrainedEmbeddingsVectorizer<T>"
description: "Converts text documents to vectors using pre-trained word embeddings loaded from files."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.TextVectorizers`

Converts text documents to vectors using pre-trained word embeddings loaded from files.

## For Beginners

Use this when you have pre-trained embeddings:

- Download pre-trained vectors (GloVe, Word2Vec, FastText) from the web
- Load them once and use for any text classification or similarity task
- Much faster than training your own embeddings
- Works great for most NLP tasks

## How It Works

This vectorizer loads pre-trained word embeddings from standard file formats
(Word2Vec text, GloVe, FastText) and uses them to create document representations.

Supported file formats:

- Word2Vec text format: First line contains vocab_size and vector_size, followed by word vectors
- GloVe format: Each line contains word followed by vector values (no header)
- FastText format: Same as Word2Vec text format

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PretrainedEmbeddingsVectorizer(Dictionary<String,Double[]>,Word2VecAggregation,Double,Boolean,Func<String,IEnumerable<String>>,HashSet<String>,ITokenizer)` | Creates a new instance from already-loaded word vectors. |
| `PretrainedEmbeddingsVectorizer(String,PretrainedFormat,Nullable<Int32>,Word2VecAggregation,Double,Boolean,Func<String,IEnumerable<String>>,HashSet<String>,ITokenizer)` | Creates a new instance of `PretrainedEmbeddingsVectorizer` by loading embeddings from a file. |

## Properties

| Property | Summary |
|:-----|:--------|
| `FeatureCount` |  |
| `IsFitted` |  |
| `VectorSize` | Gets the vector dimensionality. |
| `WordVectors` | Gets the loaded word vectors. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Fit(IEnumerable<String>)` | Fitting is not required for pre-trained embeddings vectorizer. |
| `GetWordVector(String)` | Gets the vector for a specific word. |
| `Transform(IEnumerable<String>)` | Transforms documents to dense vectors using pre-trained embeddings. |

