---
title: "RelationModule<T>"
description: "Relation module that computes similarity between feature pairs for Relation Networks."
section: "API Reference"
---

`Models & Types` · `AiDotNet.MetaLearning.Modules`

Relation module that computes similarity between feature pairs for Relation Networks.

## For Beginners

Instead of using a fixed formula to measure similarity
(like Euclidean distance), the relation module is a small neural network that LEARNS
how to compare examples. It takes two feature vectors as input (concatenated together)
and outputs a number between 0 and 1 indicating how related they are.

## How It Works

The relation module is the core component that makes Relation Networks unique.
It takes concatenated features from two examples and outputs a scalar relation score
indicating how similar/related they are.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `RelationModule(Int32)` | Initializes a new instance of RelationModule. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DefaultLossFunction` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clone` | Creates a deep copy of the relation module. |
| `DeepCopy` |  |
| `Forward(Tensor<>)` | Performs forward pass through the relation module. |
| `GetParameters` | Gets the learnable parameters of the relation module. |
| `Predict(Tensor<>)` |  |
| `SetParameters(Vector<>)` | Sets the learnable parameters of the relation module. |
| `SetTrainingMode(Boolean)` | Sets the training mode. |
| `Train(Tensor<>,Tensor<>)` |  |
| `WithParameters(Vector<>)` |  |

