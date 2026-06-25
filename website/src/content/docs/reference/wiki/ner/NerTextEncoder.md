---
title: "NerTextEncoder"
description: "Converts raw text into the packed integer indices consumed by `WordCharEmbeddingLayer` / `WordCharBiLSTMCRF`, and builds the word and character vocabularies."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NER.Preprocessing`

Converts raw text into the packed integer indices consumed by
`WordCharEmbeddingLayer` /
`WordCharBiLSTMCRF`, and builds the word and
character vocabularies. This is the data-layer counterpart to the model, mirroring the standard
NER pipeline (a separate vocabulary/tokenizer feeding a model that owns its embeddings).

## For Beginners

models work on numbers, not text. This helper assigns every word and every
letter a fixed number, splits punctuation correctly, and packs each sentence into a grid of numbers
the model can read. "Unknown" words it never saw still get a consistent number, so the model behaves
the same every run.

## How It Works

Responsibilities:

- **Tokenization:** a regex word tokenizer that splits punctuation off words, so "Paris,"

becomes ["Paris", ","] instead of an out-of-vocabulary blob — fixing the recall loss a naive
whitespace split causes.

- **Vocabulary construction:** a case-folded word vocabulary (words are lowercased for the

word-embedding lookup, matching GloVe) and a case-preserving character vocabulary (so capitalization
is captured by the character encoder, not discarded).

- **Encoding:** a sentence becomes a `[sequenceLength, 1 + maxWordLength]` integer

matrix: column 0 is the word id, the rest are the word's character ids (zero-padded).

Index 0 is the padding id and index 1 is the unknown id in both vocabularies, so unseen words/chars
map deterministically to a stable "[UNK]" embedding rather than hash-derived noise.

## Properties

| Property | Summary |
|:-----|:--------|
| `CharVocabSize` | Gets the character vocabulary size (including [PAD] and [UNK]). |
| `CharVocabulary` | Gets the character vocabulary (case-preserving). |
| `MaxWordLength` | Gets the maximum number of characters encoded per word; longer words are truncated. |
| `WordVocabSize` | Gets the word vocabulary size (including [PAD] and [UNK]). |
| `WordVocabulary` | Gets the word vocabulary (word ids are looked up case-folded). |

## Methods

| Method | Summary |
|:-----|:--------|
| `Build(IEnumerable<String[]>,Int32)` | Builds a `NerTextEncoder` from already-tokenized training sentences, populating the word and character vocabularies. |
| `EncodePacked(String[],Int32)` | Encodes a tokenized sentence into a packed `[sequenceLength, 1 + MaxWordLength]` integer matrix as `Double` values: column 0 is the (case-folded) word id, columns 1.. |
| `FromVocabularies(Vocabulary,Vocabulary,Int32)` | Reconstructs an encoder from already-built vocabularies — used when deserializing a `WordCharBiLSTMCRF` so the restored model maps tokens/characters back to the exact same embedding-row ids it was trained with. |
| `Tokenize(String)` | Splits a sentence into word/punctuation tokens, separating trailing punctuation from words. |

## Fields

| Field | Summary |
|:-----|:--------|
| `PadToken` | The padding token, reserved at index 0 in both vocabularies. |
| `UnkToken` | The unknown token, reserved at index 1 in both vocabularies. |

