---
title: "HashingVectorizer<T>"
description: "Converts text documents to a fixed-size hash-based feature matrix."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.TextVectorizers`

Converts text documents to a fixed-size hash-based feature matrix.

## For Beginners

HashingVectorizer is like CountVectorizer but:

- Doesn't need to see all data first (no vocabulary to learn)
- Uses fixed memory regardless of vocabulary size
- Slight accuracy loss from hash collisions
- Great for very large or streaming text data

## How It Works

HashingVectorizer uses the hashing trick to map tokens to a fixed number of features.
Unlike CountVectorizer and TfidfVectorizer, it doesn't require storing a vocabulary,
making it memory-efficient for very large datasets or streaming applications.

The trade-off is that hash collisions can occur (different tokens map to same feature),
and you cannot retrieve original tokens from the hash indices.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `HashingVectorizer(Int32,Nullable<ValueTuple<Int32,Int32>>,Boolean,Boolean,Boolean,HashingNorm,Func<String,IEnumerable<String>>,HashSet<String>,ITokenizer)` | Creates a new instance of `HashingVectorizer`. |

## Properties

| Property | Summary |
|:-----|:--------|
| `FeatureCount` |  |
| `IsFitted` |  |
| `NFeatures` | Gets the number of output features (hash buckets). |

## Methods

| Method | Summary |
|:-----|:--------|
| `Fit(IEnumerable<String>)` | Fits the vectorizer (no-op for HashingVectorizer - fitting is not required). |
| `FitTransform(IEnumerable<String>)` |  |
| `GetFeatureNamesOut` |  |
| `Transform(IEnumerable<String>)` | Transforms documents to hash-based feature vectors. |

