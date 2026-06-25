---
title: "WordCharBiLSTMCRF<T>"
description: "Paper-faithful word + character BiLSTM-CRF for Named Entity Recognition (Lample et al., NAACL 2016, \"Neural Architectures for Named Entity Recognition\")."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NER.SequenceLabeling`

Paper-faithful word + character BiLSTM-CRF for Named Entity Recognition
(Lample et al., NAACL 2016, "Neural Architectures for Named Entity Recognition").

## For Beginners

give this model sentences (as words) with their entity labels and it learns to
tag new sentences. It reads both whole words and their spelling, considers the whole sentence in both
directions, and makes sure the labels form valid entity spans. Use `Tensor{` to build one
from your training sentences.

## How It Works

Unlike `BiLSTMCRF`, which consumes caller-supplied token embeddings, this model
**owns its embedding layers and consumes token/character indices** — the standard NER design
(AllenNLP `CrfTagger`, flair `SequenceTagger`, the canonical PyTorch BiLSTM-CRF). The full
architecture is:

- **Word + character embedding front-end** (`WordCharEmbeddingLayer`): a

learnable word lookup (initializable from pretrained GloVe via `String)`)
concatenated with a character-level BiLSTM representation per word.

- **Word-level BiLSTM** over the per-token representations for bidirectional context.
- **Dropout** regularization (paper default 0.5).
- **Linear projection** to per-label emission scores.
- **CRF** for globally-consistent BIO decoding (Viterbi) and negative-log-likelihood training.

**Why a separate model:** owning the embeddings fixes the failure modes of feeding a BiLSTM-CRF
ad-hoc vectors: word identity gives a shared, deterministic, generalizing embedding space (no
process-randomized hashing), and the character BiLSTM lets the model recognize unseen words by their
spelling instead of hallucinating. The CRF guarantees structurally valid label sequences (no orphan
I- tags).

**References:** Lample et al., NAACL 2016; Huang, Xu, Yu, 2015; Ma and Hovy, ACL 2016.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `WordCharBiLSTMCRF(NeuralNetworkArchitecture<>,NerTextEncoder,BiLSTMCRFOptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Constructs a word+char BiLSTM-CRF over a prebuilt `NerTextEncoder`. |

## Properties

| Property | Summary |
|:-----|:--------|
| `EmbeddingFrontEnd` | Gets the word+character embedding front-end layer (null until layers are initialized). |
| `Encoder` | Gets the text encoder (word/character vocabularies + tokenizer) this model was built with. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeEmissionScores(Tensor<>)` |  |
| `Create(IEnumerable<String[]>,BiLSTMCRFOptions,Int32,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Builds a ready-to-train model from tokenized, labeled training sentences. |
| `CreateNewInstance` |  |
| `DeserializeNetworkSpecificData(BinaryReader)` |  |
| `EncodeSentence(String[])` | Encodes a tokenized sentence into the packed-index input tensor this model consumes. |
| `GetModelMetadata` |  |
| `GetOptions` |  |
| `InitializeLayers` |  |
| `LoadGloVeEmbeddings(String)` | Initializes the word-embedding table from a pretrained GloVe text file. |
| `PostprocessOutput(Tensor<>)` |  |
| `PredictLabels(Tensor<>)` |  |
| `PreprocessTokens(Tensor<>)` |  |
| `SerializeNetworkSpecificData(BinaryWriter)` |  |
| `Train(Tensor<>,Tensor<>)` |  |
| `UpdateParameters(Vector<>)` |  |
| `WarmUp` | Runs one inference pass on a zero input to resolve every lazy layer's shape and materialize its parameter tensors. |

