---
title: "RelationNetworkModel<T, TInput, TOutput>"
description: "Relation Network model for few-shot classification."
section: "API Reference"
---

`Models & Types` · `AiDotNet.MetaLearning.Models`

Relation Network model for few-shot classification.

## For Beginners

After the Relation Network sees the support examples (the few
labeled examples for each class), this model remembers them and uses them to classify
new query examples. It does this by computing how "related" the query is to each
support example.

## How It Works

This model stores the adapted state of a Relation Network after seeing support examples.
It can then classify query examples by computing relation scores with the stored support set.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `RelationNetworkModel(IFullModel<,,>,RelationModule<>,,,RelationNetworkOptions<,,>)` | Initializes a new instance of the RelationNetworkModel. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Metadata` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `AggregateScoresByClass(List<>)` | Aggregates relation scores by class. |
| `ApplySoftmax(Vector<>)` | Applies softmax to convert scores to probabilities. |
| `ComputeDifferenceFeatures(Vector<>,Vector<>)` | Computes difference features between query and support. |
| `ComputeProductFeatures(Vector<>,Vector<>)` | Computes element-wise product features between query and support. |
| `ComputeRelationScore(Vector<>,Vector<>)` | Computes the relation score between two feature vectors. |
| `ConcatenateFeatures(Vector<>,Vector<>)` | Concatenates two feature vectors into a combined tensor. |
| `ConvertToOutput(Vector<>)` | Converts probability vector to the expected output type. |
| `EncodeSample(Tensor<>)` | Encodes a tensor sample using the feature encoder. |
| `EncodeVector(Vector<>)` | Encodes a vector sample using the feature encoder. |
| `GetModelMetadata` |  |
| `GetParameters` |  |
| `PrecomputeSupportFeatures` | Pre-computes and caches support set features for efficient inference. |
| `Predict()` |  |
| `Train(,)` |  |
| `UpdateParameters(Vector<>)` |  |

