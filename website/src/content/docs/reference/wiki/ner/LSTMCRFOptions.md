---
title: "LSTMCRFOptions"
description: "Configuration options for the LSTM-CRF Named Entity Recognition model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.NER.Options`

Configuration options for the LSTM-CRF Named Entity Recognition model.

## For Beginners

LSTM-CRF is a simpler, faster version of BiLSTM-CRF that only reads
text in one direction (left to right). It's like reading a sentence without being able to
look ahead. This makes it faster for real-time applications but slightly less accurate.

Use LSTM-CRF when:

- You need lower latency for real-time NER
- You're processing streaming text (words arriving one at a time)
- You want a simpler model with fewer parameters
- Slightly lower accuracy is acceptable

## How It Works

LSTM-CRF (Huang, Xu, and Yu, 2015 - "Bidirectional LSTM-CRF Models for Sequence Tagging")
is a simpler variant that uses unidirectional LSTM with a CRF layer. While the original paper
actually proposed both unidirectional and bidirectional variants, this class implements the
unidirectional version for scenarios where lower latency or streaming inference is needed.

The architecture consists of:

1. **Unidirectional LSTM:** Processes the token sequence left-to-right only, providing

each token with context from preceding words. This enables streaming/online inference
since each token can be processed as soon as it arrives, without waiting for the full
sentence. The tradeoff is that right-context is not available, which reduces accuracy
by 1-2% F1 compared to BiLSTM-CRF.

2. **CRF decoder:** Models label transition dependencies to produce globally optimal

label sequences. The CRF is especially important for unidirectional models because it
partially compensates for the lack of right-context by leveraging label sequence patterns.

Default values:

- 100-dimensional word embeddings
- Single LSTM layer with 100 hidden units
- 50% dropout rate
- 9 CoNLL-2003 BIO labels

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LSTMCRFOptions` | Initializes a new instance with default values. |
| `LSTMCRFOptions(LSTMCRFOptions)` | Initializes a new instance by deep-copying all settings from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DropoutRate` | Gets or sets the dropout rate for regularization. |
| `EmbeddingDimension` | Gets or sets the input token embedding dimension. |
| `HiddenDimension` | Gets or sets the LSTM hidden state dimension. |
| `LabelNames` | Gets or sets the BIO label names. |
| `LearningRate` | Gets or sets the learning rate for the optimizer. |
| `MaxSequenceLength` | Gets or sets the maximum input sequence length in tokens. |
| `ModelPath` |  |
| `NumLSTMLayers` | Gets or sets the number of stacked LSTM layers. |
| `NumLabels` | Gets or sets the number of entity label classes in the BIO tagging scheme. |
| `OnnxOptions` |  |
| `UseCRF` | Gets or sets whether to use CRF decoding. |
| `Variant` |  |

