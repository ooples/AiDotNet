---
title: "SequenceLabelingNERBase<T>"
description: "Base class for sequence labeling NER models that assign a BIO label to each token in a sequence."
section: "API Reference"
---

`Base Classes` · `AiDotNet.NER.SequenceLabeling`

Base class for sequence labeling NER models that assign a BIO label to each token in a sequence.

## For Beginners

Sequence labeling NER processes text one word at a time and assigns
a label to each word. The labels use a special coding scheme called BIO:

- **B** = "Begin" - marks the first word of an entity
- **I** = "Inside" - marks continuation words of an entity
- **O** = "Outside" - marks words that aren't part of any entity

For example, in "John Smith works at Google":

- "John" -> B-PER (Beginning of a Person name)
- "Smith" -> I-PER (Inside the same Person name)
- "works" -> O (not an entity)
- "at" -> O (not an entity)
- "Google" -> B-ORG (Beginning of an Organization name)

The key advantage of this approach is that it naturally handles multi-word entities
like "New York City" (B-LOC, I-LOC, I-LOC).

## How It Works

Sequence labeling is the most common approach to Named Entity Recognition, where each token
in a sentence is assigned a label from a predefined set using the BIO (Begin, Inside, Outside)
tagging scheme. This base class provides the task-specific functionality shared by all
sequence labeling NER models, analogous to how `VideoSuperResolutionBase<T>`
provides shared functionality for all video super-resolution models.

The BIO scheme works as follows:

- **B-TYPE:** Beginning of an entity of the given type (e.g., B-PER for the first token of a person name)
- **I-TYPE:** Inside (continuation of) an entity (e.g., I-PER for subsequent tokens of a person name)
- **O:** Outside any entity (regular words like verbs, prepositions, articles)

For example: "Barack Obama was born in Honolulu"

- Barack -> B-PER (beginning of a person entity)
- Obama -> I-PER (inside the same person entity)
- was -> O (not an entity)
- born -> O (not an entity)
- in -> O (not an entity)
- Honolulu -> B-LOC (beginning of a location entity)

This base class provides:

- Label sequence prediction (abstract, implemented by concrete models)
- Emission score computation (the per-token, per-label scores before CRF decoding)
- Argmax decoding fallback for models without CRF
- Label index to label name conversion utilities
- CRF toggle for enabling/disabling structured prediction

Derived classes implement specific architectures like BiLSTM-CRF, BERT-NER, SpanNER, etc.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SequenceLabelingNERBase(NeuralNetworkArchitecture<>,ILossFunction<>,Double)` | Initializes a new instance of the SequenceLabelingNERBase class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `LabelNames` | Gets the label names corresponding to each label index in the BIO tagging scheme. |
| `UseCRF` | Gets or sets whether to use CRF (Conditional Random Field) decoding for label sequence prediction. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ArgmaxDecode(Tensor<>)` | Performs independent argmax decoding on emission scores to get label predictions. |
| `ComputeEmissionScores(Tensor<>)` | Computes emission scores from token embeddings without CRF decoding. |
| `DecodeLabels(Tensor<>)` | Converts predicted label indices to human-readable BIO label name strings. |
| `DecodeLabelsBatch(Tensor<>)` | Converts batched predicted label indices to human-readable BIO label name strings. |
| `FindCrfLayer` | Returns the CRF layer in the model's Layers list, or null if absent (e.g. |
| `PredictCore(Tensor<>)` |  |
| `PredictLabels(Tensor<>)` | Predicts the optimal BIO label sequence for input token embeddings. |
| `PreprocessLabels(Tensor<>,Int32)` | Pads or truncates a labels tensor along the sequence axis so its length matches the CRF's locked sequence length (`targetSeqLen`). |
| `RunCrfAwareTrainStep(Tensor<>,Tensor<>,Boolean,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Runs one CRF-aware (when `useCrf` is true and a CRF layer is present) or cross-entropy training step. |

