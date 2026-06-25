---
title: "ITextVectorizer<T>"
description: "Defines a text vectorizer that converts text documents to numeric feature matrices."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Defines a text vectorizer that converts text documents to numeric feature matrices.

## For Beginners

A text vectorizer converts words into numbers that ML models can understand.
Different vectorizers use different strategies:

- TF-IDF: Weights words by importance (rare words score higher)
- Count: Simply counts how many times each word appears
- Hashing: Uses hashing for memory efficiency with large vocabularies
- BM25: Improved TF-IDF used by search engines
- Word2Vec/Doc2Vec: Creates dense embeddings capturing semantic meaning

## How It Works

Text vectorizers transform collections of text documents into numeric representations
suitable for machine learning algorithms. They follow the sklearn-style Fit/Transform
pattern where the vectorizer first learns from training data, then transforms any text.

## Properties

| Property | Summary |
|:-----|:--------|
| `FeatureCount` | Gets the number of features (vocabulary size) this vectorizer produces. |
| `FeatureNames` | Gets the feature names (vocabulary terms in order). |
| `IsFitted` | Gets whether this vectorizer has been fitted to data. |
| `Vocabulary` | Gets the vocabulary mapping (token to index). |

## Methods

| Method | Summary |
|:-----|:--------|
| `Fit(IEnumerable<String>)` | Fits the vectorizer to the training documents, learning the vocabulary. |
| `FitTransform(IEnumerable<String>)` | Fits the vectorizer and transforms the documents in one step. |
| `GetFeatureNamesOut` | Gets the output feature names. |
| `Transform(IEnumerable<String>)` | Transforms documents to a numeric feature matrix. |

