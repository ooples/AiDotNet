---
title: "TfidfVectorizer<T>"
description: "Converts a collection of text documents to a TF-IDF weighted matrix."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.TextVectorizers`

Converts a collection of text documents to a TF-IDF weighted matrix.

## For Beginners

TF-IDF makes rare but meaningful words more important:

- Common words like "the" appear everywhere, so they get low weight
- Rare words that only appear in specific documents get high weight
- This helps distinguish documents by their unique content

Example: "machine learning" in a tech article is more meaningful
than "the" which appears in every document.

## How It Works

TfidfVectorizer converts text to TF-IDF (Term Frequency-Inverse Document Frequency)
representation, which weights terms by their importance in a document relative
to the entire corpus.

TF-IDF = TF × IDF where:

- TF (Term Frequency) = count of term in document / total terms in document
- IDF (Inverse Document Frequency) = log(total documents / documents containing term)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TfidfVectorizer(Int32,Double,Nullable<Int32>,Nullable<ValueTuple<Int32,Int32>>,Boolean,TfidfNorm,Boolean,Boolean,Boolean,Func<String,IEnumerable<String>>,HashSet<String>,ITokenizer)` | Creates a new instance of `TfidfVectorizer`. |

## Properties

| Property | Summary |
|:-----|:--------|
| `IdfWeights` | Gets the IDF weights for each term. |
| `IsFitted` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Fit(IEnumerable<String>)` | Fits the vectorizer to the corpus. |
| `Transform(IEnumerable<String>)` | Transforms documents to TF-IDF weighted vectors. |

