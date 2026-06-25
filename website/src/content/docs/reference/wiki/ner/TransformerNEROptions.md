---
title: "TransformerNEROptions"
description: "Base configuration options shared by all transformer-based NER models (BERT-NER, RoBERTa-NER, DeBERTa-NER, ELECTRA-NER, etc.)."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.NER.Options`

Base configuration options shared by all transformer-based NER models (BERT-NER, RoBERTa-NER,
DeBERTa-NER, ELECTRA-NER, etc.).

## For Beginners

Transformer models are the most powerful NER architectures available.
They read text using "self-attention" - each word looks at every other word in the sentence
to understand context. This is more powerful than BiLSTM-CRF because transformers can
capture long-range dependencies (e.g., a pronoun referring to an entity mentioned 50 words ago).

The tradeoff is that transformer models are much larger (110M+ parameters for BERT-base vs
~1M for BiLSTM-CRF) and slower to train. However, they achieve state-of-the-art accuracy
on all NER benchmarks.

## How It Works

Transformer-based NER models share a common architecture: a pre-trained transformer encoder
produces contextualized token representations, followed by a token classification head that
predicts BIO labels for each token. The key differences between variants are in the
pre-training strategy and attention mechanism, not the NER fine-tuning architecture.

The standard transformer NER architecture consists of:

1. **Transformer encoder:** Multi-layer self-attention with feed-forward networks.

Each layer has multi-head attention (captures token-to-token relationships) and a
feed-forward network (transforms each token's representation independently).

2. **Classification head:** A linear projection from hidden_size to num_labels,

optionally with a CRF layer for structured prediction.

3. **Optional CRF:** Can be added on top of the transformer for structured decoding.

This base options class provides the shared parameters. Variant-specific options classes
(BERTNEROptions, RoBERTaNEROptions, etc.) can extend this with model-specific settings.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TransformerNEROptions` | Initializes a new instance with BERT-base defaults. |
| `TransformerNEROptions(TransformerNEROptions)` | Initializes a new instance by deep-copying all settings from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DropoutRate` | Gets or sets the dropout rate for regularization. |
| `HiddenDimension` | Gets or sets the hidden dimension of the transformer encoder. |
| `IntermediateDimension` | Gets or sets the intermediate (feed-forward) dimension in each transformer layer. |
| `LabelNames` | Gets or sets the BIO label names. |
| `LearningRate` | Gets or sets the learning rate for fine-tuning. |
| `MaxSequenceLength` | Gets or sets the maximum input sequence length in tokens. |
| `ModelPath` | Gets or sets the path to a pre-trained ONNX model file. |
| `NumAttentionHeads` | Gets or sets the number of self-attention heads in each transformer layer. |
| `NumLabels` | Gets or sets the number of entity label classes in the BIO tagging scheme. |
| `NumTransformerLayers` | Gets or sets the number of transformer encoder layers. |
| `OnnxOptions` | Gets or sets the ONNX Runtime configuration options. |
| `UseCRF` | Gets or sets whether to use CRF decoding on top of the transformer. |
| `Variant` | Gets or sets the model size variant. |

