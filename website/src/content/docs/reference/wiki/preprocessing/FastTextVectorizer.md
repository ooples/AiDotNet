---
title: "FastTextVectorizer<T>"
description: "Converts text documents to dense vectors using FastText-style subword embeddings."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.TextVectorizers`

Converts text documents to dense vectors using FastText-style subword embeddings.

## For Beginners

FastText is Word2Vec that understands word parts:

- Can generate vectors for words it has never seen before
- "unhappiness" can be understood from "un-", "happy", "-ness" patterns
- Better for languages with rich morphology (German, Turkish, Finnish)
- Handles typos and spelling variations better than Word2Vec

## How It Works

FastText extends Word2Vec by representing each word as a bag of character n-grams.
This allows the model to:

- Handle out-of-vocabulary (OOV) words by composing subword vectors
- Capture morphological information (prefixes, suffixes, roots)
- Be more robust to spelling variations and typos

For example, "where" with n-grams (3,6) includes: "<wh", "whe", "her", "ere", "re>",
"<whe", "wher", "here", "ere>", "<wher", "where", "here>", "<where", "where>", "<where>"

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FastTextVectorizer(Int32,Int32,Int32,Int32,Double,Int32,Nullable<ValueTuple<Int32,Int32>>,Int32,Word2VecAggregation,Boolean,Nullable<Int32>,Func<String,IEnumerable<String>>,HashSet<String>,ITokenizer)` | Creates a new instance of `FastTextVectorizer`. |

## Properties

| Property | Summary |
|:-----|:--------|
| `FeatureCount` |  |
| `IsFitted` |  |
| `VectorSize` | Gets the vector dimensionality. |
| `WordVectors` | Gets the learned word vectors for known words. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Fit(IEnumerable<String>)` | Trains FastText embeddings on the corpus. |
| `GetSubwordHash(String)` | Gets the hash bucket index for a subword. |
| `GetSubwords(String)` | Gets the character n-grams for a word. |
| `GetWordVector(String)` | Gets the vector for a word, computing from subwords if unknown. |
| `Transform(IEnumerable<String>)` | Transforms documents to dense vectors by aggregating word vectors. |

