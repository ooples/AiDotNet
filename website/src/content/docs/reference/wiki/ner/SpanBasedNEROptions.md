---
title: "SpanBasedNEROptions"
description: "Options shared by span-based NER models (SpERT, BiaffineNER, PURE)."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.NER.Options`

Options shared by span-based NER models (SpERT, BiaffineNER, PURE).

## How It Works

Span-based NER models enumerate all possible spans in a sentence and classify each span
as an entity type or non-entity. This contrasts with sequence labeling (BIO tagging) where
each token is independently labeled. Span-based models handle nested entities naturally
because overlapping spans can each have different labels.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SpanBasedNEROptions` | Creates a new instance with default settings (BERT-base defaults). |
| `SpanBasedNEROptions(SpanBasedNEROptions)` | Deep-copy constructor. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DropoutRate` | Gets or sets the dropout rate. |
| `HiddenDimension` | Gets or sets the hidden dimension of the encoder. |
| `IntermediateDimension` | Gets or sets the feed-forward intermediate dimension. |
| `LabelNames` | Gets or sets the label names for the NER tags. |
| `LearningRate` | Gets or sets the learning rate for the optimizer. |
| `MaxSequenceLength` | Gets or sets the maximum sequence length in tokens. |
| `MaxSpanLength` | Gets or sets the maximum span length (number of tokens in a single entity span). |
| `ModelPath` | Gets or sets the path to an ONNX model file for inference mode. |
| `NegativeSpanSampleRatio` | Gets or sets the ratio of negative (non-entity) spans to positive (entity) spans sampled during training. |
| `NumAttentionHeads` | Gets or sets the number of attention heads in each transformer layer. |
| `NumLabels` | Gets or sets the number of entity labels (including O tag). |
| `NumTransformerLayers` | Gets or sets the number of transformer encoder layers. |
| `OnnxOptions` | Gets or sets ONNX runtime inference options. |
| `SpanEmbeddingDimension` | Gets or sets the span embedding dimension used for span classification. |
| `Variant` | Gets or sets the NER model variant. |

