---
title: "MetaLearningModelBase<T, TInput, TOutput>"
description: "Abstract base class for meta-learning adapted models that wrap a base model with task-specific parameters."
section: "API Reference"
---

`Base Classes` · `AiDotNet.MetaLearning.Models`

Abstract base class for meta-learning adapted models that wrap a base model with task-specific parameters.

## For Beginners

Meta-learning adapted models are created by meta-learning algorithms
(like MAML, ProtoNets, etc.) after adapting to a new task. They wrap a base neural network
with task-specific parameters. This base class handles common concerns like serialization,
gradient computation, and feature awareness by delegating to the wrapped model.

## How It Works

Extends `ModelWrapperBase` with meta-learning-specific behavior:
training is not supported directly (use the meta-learning algorithm instead), and parameter
management is abstract so each adapted model can store its own task-specific parameters.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MetaLearningModelBase(IFullModel<,,>)` | Initializes a new instance of the `MetaLearningModelBase` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ParameterCount` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyGradients(Vector<>,)` |  |
| `ConvertVectorToOutput(Vector<>)` | Helper method to convert a vector output to the expected TOutput type. |
| `ExtractFeaturesFromBaseModel(,Int32)` | Helper method to extract features from the base model output as a vector. |
| `GetParameters` |  |
| `SetParameters(Vector<>)` |  |
| `Train(,)` |  |

