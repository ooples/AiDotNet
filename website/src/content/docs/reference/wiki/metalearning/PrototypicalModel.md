---
title: "PrototypicalModel<T, TInput, TOutput>"
description: "Prototypical model for few-shot classification."
section: "API Reference"
---

`Models & Types` · `AiDotNet.MetaLearning.Algorithms`

Prototypical model for few-shot classification.

## For Beginners

After adapting ProtoNets to a new task, you get this model.
It can classify new examples instantly by finding the nearest class prototype.

## How It Works

This model encapsulates the ProtoNets inference mechanism with pre-computed prototypes.
It is returned by `IMetaLearningTask{` and provides
fast classification without any gradient computation.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PrototypicalModel(IFullModel<,,>,,,ProtoNetsOptions<,,>,INumericOperations<>)` | Initializes a new instance of the PrototypicalModel. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Metadata` | Gets the model metadata. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputePrototypes(,)` | Computes class prototypes from support set. |
| `GetModelMetadata` | Gets metadata about the model. |
| `GetParameters` | Gets model parameters (not applicable for prototype-based models). |
| `Predict()` | Makes predictions using prototype-based classification. |
| `Train(,)` | Trains the model (not applicable for prototype-based models). |
| `UpdateParameters(Vector<>)` | Updates model parameters (not applicable for prototype-based models). |

