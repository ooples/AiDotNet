---
title: "Doc2VecVectorizer<T>"
description: "Converts text documents to dense vectors using Doc2Vec (Paragraph Vector)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.TextVectorizers`

Converts text documents to dense vectors using Doc2Vec (Paragraph Vector).

## For Beginners

Doc2Vec is Word2Vec for whole documents:

- Each document gets its own unique learned vector
- Captures document meaning, not just word averages
- Great for document similarity, classification, and clustering
- Better than averaging word vectors for longer documents

## How It Works

Doc2Vec extends Word2Vec to learn fixed-length representations for documents.
Two architectures are supported:

- PV-DM (Distributed Memory): Uses document vector with context words to predict target
- PV-DBOW (Distributed Bag of Words): Uses document vector alone to predict words

Unlike Word2Vec averaging, Doc2Vec learns document-specific vectors that capture
document-level semantics beyond just word averages.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `Doc2VecVectorizer(Int32,Int32,Int32,Int32,Double,Int32,Doc2VecArchitecture,Boolean,Nullable<Int32>,Func<String,IEnumerable<String>>,HashSet<String>,ITokenizer)` | Creates a new instance of `Doc2VecVectorizer`. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DocVectors` | Gets the learned document vectors (indexed by training order). |
| `FeatureCount` |  |
| `IsFitted` |  |
| `VectorSize` | Gets the vector dimensionality. |
| `WordVectors` | Gets the learned word vectors. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Fit(IEnumerable<String>)` | Trains Doc2Vec embeddings on the corpus. |
| `Transform(IEnumerable<String>)` | Transforms documents to dense vectors by inferring document vectors. |

