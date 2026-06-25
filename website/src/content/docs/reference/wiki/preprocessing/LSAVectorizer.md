---
title: "LSAVectorizer<T>"
description: "Converts text documents to latent semantic vectors using Latent Semantic Analysis (LSA/LSI)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.TextVectorizers`

Converts text documents to latent semantic vectors using Latent Semantic Analysis (LSA/LSI).

## For Beginners

LSA discovers hidden topics in your text:

- "car" and "automobile" become similar because they appear in similar contexts
- Reduces thousands of word features to ~100-500 semantic concepts
- Great for document similarity, clustering, and information retrieval
- Can handle synonymy (different words, same meaning) and polysemy (same word, different meanings)

## How It Works

LSA (Latent Semantic Analysis), also known as LSI (Latent Semantic Indexing), uses
Singular Value Decomposition (SVD) to reduce the dimensionality of the term-document
matrix while capturing latent semantic relationships between terms and documents.

The algorithm works by:

1. Creating a TF-IDF weighted term-document matrix
2. Applying truncated SVD to get a low-rank approximation
3. Projecting documents into the reduced semantic space

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LSAVectorizer(Int32,Int32,Double,Int32,Double,Nullable<Int32>,Nullable<ValueTuple<Int32,Int32>>,Boolean,Nullable<Int32>,Func<String,IEnumerable<String>>,HashSet<String>,ITokenizer)` | Creates a new instance of `LSAVectorizer`. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Components` | Gets the components (topic-term matrix) from LSA. |
| `ExplainedVarianceRatio` | Gets the explained variance ratio for each component. |
| `FeatureCount` |  |
| `IsFitted` |  |
| `NComponents` | Gets the number of components (latent dimensions). |
| `SingularValues` | Gets the singular values from the SVD decomposition. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Fit(IEnumerable<String>)` | Fits the LSA vectorizer to the corpus. |
| `PerformTruncatedSVD(Double[0:,0:],Int32)` | Performs truncated SVD using randomized algorithm. |
| `Transform(IEnumerable<String>)` | Transforms documents to LSA vectors. |

