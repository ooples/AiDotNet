---
title: "CNNBiLSTMCRFOptions"
description: "Configuration options for the CNN-BiLSTM-CRF Named Entity Recognition model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.NER.Options`

Configuration options for the CNN-BiLSTM-CRF Named Entity Recognition model.

## For Beginners

CNN-BiLSTM-CRF is like BiLSTM-CRF but with an extra component that
looks at the letters within each word. A CNN (Convolutional Neural Network) slides a small
window across the characters of each word to detect patterns like capitalization ("John" vs
"john"), suffixes ("-tion", "-ing"), and word shapes. These character features are combined
with word embeddings before being fed into the BiLSTM.

Compared to BiLSTM-CRF with character BiLSTM:

- CNN is faster to compute (parallel vs sequential processing)
- CNN is better at capturing local character n-grams (suffixes, prefixes)
- BiLSTM is better at capturing long-range character dependencies (rare)
- In practice, both achieve similar NER accuracy (~91% F1 on CoNLL-2003)

## How It Works

CNN-BiLSTM-CRF (Ma and Hovy, ACL 2016 - "End-to-end Sequence Labeling via Bi-directional
LSTM-CNNs-CRF") extends the BiLSTM-CRF architecture with character-level Convolutional Neural
Network (CNN) embeddings. The architecture has three main components:

1. **Character-level CNN:** A 1D CNN processes each word's character sequence to capture

morphological features (capitalization, prefixes, suffixes, word shape). Unlike the
character BiLSTM in Lample et al. (2016), a CNN is faster and better at capturing local
n-gram patterns. The CNN output is max-pooled to produce a fixed-size vector per word.

2. **BiLSTM encoder:** Processes the concatenation of word embeddings and character CNN

features in both forward and backward directions. Each token gets a representation
that captures its full sentence context plus its morphological features.

3. **CRF decoder:** Models label transition dependencies to produce globally optimal

label sequences, enforcing valid BIO constraints via the Viterbi algorithm.

Default values follow the original Ma and Hovy (2016) paper:

- 100-dimensional GloVe word embeddings
- Character CNN with 30-dimensional embeddings, 30 filters of width 3
- Single BiLSTM layer with 200 hidden units per direction
- 50% dropout rate
- 9 CoNLL-2003 BIO labels

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CNNBiLSTMCRFOptions` | Initializes a new instance with default values matching the original Ma and Hovy (2016) paper. |
| `CNNBiLSTMCRFOptions(CNNBiLSTMCRFOptions)` | Initializes a new instance by deep-copying all settings from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `CharCNNFilters` | Gets or sets the number of CNN filters for character-level feature extraction. |
| `CharCNNKernelSize` | Gets or sets the kernel (window) size for the character CNN. |
| `CharEmbeddingDimension` | Gets or sets the character embedding vector dimension. |
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
| `Variant` | Gets or sets the model size variant, which controls the overall capacity of the model. |

