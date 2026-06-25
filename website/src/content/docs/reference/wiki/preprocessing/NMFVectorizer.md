---
title: "NMFVectorizer<T>"
description: "Converts text documents to topic vectors using Non-negative Matrix Factorization (NMF)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.TextVectorizers`

Converts text documents to topic vectors using Non-negative Matrix Factorization (NMF).

## For Beginners

NMF is like LSA but with only positive values:

- Topics are "built from" words (additive combinations)
- Results are often more interpretable than LSA
- Works well for topic modeling when you want to "see" what makes up each topic
- Each document is a combination of topics with positive weights

## How It Works

NMF factorizes the term-document matrix V into two non-negative matrices W and H such that V ≈ W × H.

- W: document-topic matrix (n_docs × n_topics)
- H: topic-term matrix (n_topics × n_terms)

Unlike LSA which can have negative values, NMF produces purely additive, parts-based representations
that are often more interpretable. Topics are combinations of words with positive weights.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `NMFVectorizer(Int32,Int32,Double,NMFInitialization,Int32,Double,Nullable<Int32>,Nullable<ValueTuple<Int32,Int32>>,Boolean,Nullable<Int32>,Func<String,IEnumerable<String>>,HashSet<String>,ITokenizer)` | Creates a new instance of `NMFVectorizer`. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Components` | Gets the topic-term matrix (H) from NMF. |
| `FeatureCount` |  |
| `IsFitted` |  |
| `NComponents` | Gets the number of components (topics). |

## Methods

| Method | Summary |
|:-----|:--------|
| `Fit(IEnumerable<String>)` | Fits the NMF model to the corpus using multiplicative update rules. |
| `GetTopWordsPerTopic(Int32)` | Gets the top words for each topic. |
| `InitializeNNDSVD(Double[0:,0:],Double[0:,0:],Double[0:,0:],Int32,Int32,Int32)` | NNDSVD-inspired initialization using scaled random values. |
| `Transform(IEnumerable<String>)` | Transforms documents to NMF topic vectors. |

