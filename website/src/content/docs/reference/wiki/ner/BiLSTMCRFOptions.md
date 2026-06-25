---
title: "BiLSTMCRFOptions"
description: "Configuration options for the BiLSTM-CRF Named Entity Recognition model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.NER.Options`

Configuration options for the BiLSTM-CRF Named Entity Recognition model.

## For Beginners

BiLSTM-CRF is the most widely-used neural NER architecture. It reads
text forwards and backwards to understand each word in context, then uses a CRF to pick the
best sequence of entity labels. These options let you configure the model's size, what
embeddings it expects, and how it trains.

The default settings are a good starting point for most NER tasks. If you need higher accuracy,
try increasing HiddenDimension to 256 or using 300-dimensional GloVe embeddings.

## How It Works

BiLSTM-CRF (Huang et al., 2015; Lample et al., NAACL 2016) combines bidirectional LSTM
with a Conditional Random Field layer for sequence labeling. The architecture consists of:

1. **Word embeddings:** Pre-trained vectors (GloVe, Word2Vec) map words to dense vectors

that capture semantic meaning. Words with similar meanings have similar vectors.

2. **BiLSTM encoder:** Processes the word embeddings in both forward and backward directions.

The forward LSTM reads left-to-right (capturing "John works at ...") while the backward LSTM
reads right-to-left (capturing "... at Google Inc."). Their outputs are merged to give each
token a representation informed by its full sentence context.

3. **CRF decoder:** Models label transition dependencies. Instead of classifying each token

independently, the CRF finds the globally optimal label sequence using the Viterbi algorithm.
This ensures valid BIO transitions (e.g., I-PER can only follow B-PER or I-PER).

Default values follow the original paper (Lample et al., NAACL 2016):

- 100-dimensional GloVe word embeddings
- Single BiLSTM layer with 100 hidden units per direction
- 50% dropout rate
- 9 CoNLL-2003 BIO labels

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `BiLSTMCRFOptions` | Initializes a new instance with default values matching the original research paper. |
| `BiLSTMCRFOptions(BiLSTMCRFOptions)` | Initializes a new instance by deep-copying all settings from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `CharEmbeddingDimension` | Gets or sets the character embedding vector dimension. |
| `CharHiddenDimension` | Gets or sets the hidden dimension of the character-level LSTM. |
| `DropoutRate` | Gets or sets the dropout rate for regularization during training. |
| `EmbeddingDimension` | Gets or sets the input token embedding dimension. |
| `HiddenDimension` | Gets or sets the LSTM hidden state dimension per direction. |
| `LabelNames` | Gets or sets the BIO label names for the tagging scheme. |
| `LearningRate` | Gets or sets the learning rate for the optimizer during training. |
| `MaxSequenceLength` | Gets or sets the maximum input sequence length in tokens. |
| `ModelPath` | Gets or sets the path to a pre-trained ONNX model file for inference mode. |
| `NumLSTMLayers` | Gets or sets the number of stacked BiLSTM layers. |
| `NumLabels` | Gets or sets the number of entity label classes in the BIO tagging scheme. |
| `OnnxOptions` | Gets or sets the ONNX Runtime configuration options. |
| `UseCRF` | Gets or sets whether to use CRF (Conditional Random Field) decoding. |
| `UseCharEmbeddings` | Gets or sets whether to use character-level embeddings. |
| `Variant` | Gets or sets the model size variant, which controls the overall capacity of the model. |

