---
title: "StaticWordEmbeddingModel<T>"
description: "Implements a static word embedding model (e.g., GloVe, Word2Vec, FastText)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.RetrievalAugmentedGeneration.Embeddings`

Implements a static word embedding model (e.g., GloVe, Word2Vec, FastText).

## For Beginners

Before Transformers (like BERT), we used "static" embeddings.

- "Static" means a word always has the same vector, regardless of context.
- "bank" in "river bank" and "bank account" gets the same vector.
- Transformer models generate "contextual" embeddings where "bank" would differ based on the sentence.

Despite being older, these models are:

- Very fast (simple lookup + average)
- Low memory (if vocabulary is pruned)
- Good baselines for simple tasks

## How It Works

This model loads pre-trained word vectors from a file (standard text format where each line is "word val1 val2 ...")
and computes sentence embeddings by averaging the vectors of the words in the sentence.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `StaticWordEmbeddingModel(Dictionary<String,Vector<>>,Int32,Boolean)` | Initializes a new instance of the `StaticWordEmbeddingModel` class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `LoadFromTextFile(String,Nullable<Int32>)` | Loads embeddings from a standard text format file (GloVe/FastText format). |

