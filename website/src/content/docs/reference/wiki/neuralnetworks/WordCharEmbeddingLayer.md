---
title: "WordCharEmbeddingLayer<T>"
description: "Paper-faithful word + character embedding front-end for sequence-labeling NER models (Lample et al., NAACL 2016, \"Neural Architectures for Named Entity Recognition\", §3)."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers`

Paper-faithful word + character embedding front-end for sequence-labeling NER models
(Lample et al., NAACL 2016, "Neural Architectures for Named Entity Recognition", §3).

## For Beginners

a network can't read text directly — words must become numbers. This layer is
the "reading" front-end: it looks up a learned vector for each word AND spells each word out
letter-by-letter through a tiny LSTM, so the model can also guess at words it never saw. It glues the
two together for every word and hands the result to the rest of the network.

## How It Works

This composite layer turns a sentence of integer token/character indices into the dense
per-token representation that the downstream word-level BiLSTM-CRF consumes. It implements
the two complementary embedding streams from Lample et al. (2016):

- **Word embeddings:** a learnable lookup table (initializable from pretrained GloVe

vectors) maps each word index to a dense vector — word identity and, when pretrained, semantic
similarity.

- **Character embeddings:** a small bidirectional LSTM reads each word's character

sequence and produces a fixed-size vector per word — morphology (capitalization, prefixes/suffixes,
the shape of out-of-vocabulary words) that the word lookup alone cannot represent.

The two streams are concatenated per token, exactly as in the paper, giving a representation
of size `wordEmbeddingDim + charHiddenDim`.

**Implementation note — embeddings as linear layers.** An embedding lookup is mathematically a
linear projection of a one-hot index vector, so each lookup table is implemented as a no-frills
`DenseLayer` applied to a one-hot encoding. This is deliberate: it routes the table
through the same gradient-tape-tracked matmul that trains every other dense layer, so the word and
character tables actually learn end-to-end. (A raw integer gather is faster for large vocabularies
but, in this engine's eager training path, does not propagate gradients back to the table — which
would leave the embeddings frozen at their random initialization.) For very large vocabularies the
one-hot matmul costs more memory than a gather; for typical NER vocabularies it is fine.

**Why this matters:** feeding a BiLSTM-CRF pre-computed, identity-only embeddings (or hash-derived
vectors) yields a model that can only memorize the training vocabulary, hallucinates on unseen words,
and — with a process-randomized hash — produces non-deterministic output. Owning real learnable word
and character tables is the standard fix.

**Input contract.** A single packed integer tensor of shape `[sequenceLength, 1 + maxWordLength]`
per sentence: column 0 holds the word index, columns 1..maxWordLength hold that word's character
indices (zero-padded). Index 0 is reserved for padding. Output shape is
`[sequenceLength, wordEmbeddingDim + charHiddenDim]`.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `WordCharEmbeddingLayer(Int32,Int32,Int32,Int32,Int32,Int32,Int32)` | Initializes a new `WordCharEmbeddingLayer`. |

## Properties

| Property | Summary |
|:-----|:--------|
| `OutputEmbeddingDim` | Gets the dimensionality of the per-token output vector (`wordEmbeddingDim + charHiddenDim`) that feeds the downstream word-level BiLSTM. |
| `SupportsTraining` |  |
| `WordEmbedding` | Gets the underlying word-embedding linear layer, exposed so a model can initialize it from pretrained vectors. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Forward(Tensor<>)` | Runs the word + character embedding front-end. |
| `GetParameters` |  |
| `ResetState` |  |
| `SetParameters(Vector<>)` |  |
| `UpdateParameters()` |  |

