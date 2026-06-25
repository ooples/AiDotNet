---
title: "TextVectorizerBase<T>"
description: "Base class for text vectorizers providing common functionality."
section: "API Reference"
---

`Base Classes` · `AiDotNet.Preprocessing.TextVectorizers`

Base class for text vectorizers providing common functionality.

## For Beginners

This is the foundation that all text vectorizers build upon.
It handles the common task of breaking text into tokens and managing which words
the vectorizer knows about. Specific vectorizers (TF-IDF, Count, etc.) add their
own logic for how to convert those tokens into numbers.

## How It Works

This base class provides shared implementation for text vectorization including:

- Tokenization (splitting text into words/tokens)
- N-gram generation (creating word sequences)
- Stop word filtering
- Vocabulary management

Supports two tokenization approaches:

- **Simple Tokenizer (default):** Uses word-level tokenization via a custom function

or the built-in whitespace/punctuation splitter. Best for traditional NLP tasks.

- **ITokenizer Integration:** Uses advanced subword tokenization (BPE, WordPiece, SentencePiece)

from the AiDotNet.Tokenization module. Enables consistent tokenization between classical ML
and neural network approaches.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TextVectorizerBase(Int32,Double,Nullable<Int32>,Nullable<ValueTuple<Int32,Int32>>,Boolean,Func<String,IEnumerable<String>>,HashSet<String>,ITokenizer)` | Initializes a new instance of the text vectorizer base class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `EnglishStopWords` | Common English stop words. |
| `FeatureCount` |  |
| `FeatureNames` |  |
| `IsFitted` |  |
| `Vocabulary` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `BuildVocabulary(Dictionary<String,Int32>,Dictionary<String,Int32>)` | Builds vocabulary from document frequency counts. |
| `Fit(IEnumerable<String>)` |  |
| `FitTransform(IEnumerable<String>)` |  |
| `GenerateNGrams(IEnumerable<String>)` | Generates n-grams from a sequence of tokens. |
| `GetFeatureNamesOut` |  |
| `Tokenize(String)` | Tokenizes a document into individual tokens. |
| `Transform(IEnumerable<String>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `NumOps` | Numeric operations helper for type T. |
| `_advancedTokenizer` | Advanced tokenizer implementing ITokenizer (BPE, WordPiece, SentencePiece). |
| `_featureNames` | The feature names (tokens in order). |
| `_lowercase` | Whether to convert text to lowercase. |
| `_maxDf` | Maximum document frequency (proportion 0-1). |
| `_maxFeatures` | Maximum number of features (vocabulary size limit). |
| `_minDf` | Minimum document frequency (absolute count). |
| `_nDocs` | Number of documents seen during fitting. |
| `_nGramRange` | N-gram range (min, max) for token generation. |
| `_stopWords` | Set of stop words to exclude. |
| `_tokenizer` | Custom tokenizer function (simple word-level tokenization). |
| `_vocabulary` | The learned vocabulary (token to index mapping). |

