---
title: "ModelHelper<T, TInput, TOutput>"
description: "Provides helper methods for model-related operations."
section: "API Reference"
---

`Helpers & Utilities` · `AiDotNet.Helpers`

Provides helper methods for model-related operations.

## For Beginners

This helper class contains methods for creating and working with different
types of machine learning models. It makes it easier to initialize models, handle different data types,
and perform common operations needed when working with models.

## Properties

| Property | Summary |
|:-----|:--------|
| `_random` | Gets the thread-safe random number generator for creating randomized models. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateDefaultModel` | Creates a default implementation of IFullModel based on the input and output types. |
| `CreateDefaultModelData` | Creates default empty model data for initialization purposes. |
| `CreateRandomExpressionTree(Int32)` | Creates a random expression tree with a specified maximum depth. |
| `CreateRandomExpressionTreeWithFeatures(Int32[],Int32)` | Creates a random expression tree that uses only the specified feature indices. |
| `CreateRandomModelWithFeatures(Int32[],Int32,Boolean,Int32)` | Creates a random model that emphasizes specific features. |
| `CreateRandomNeuralNetworkWithFeatures(Int32[],Int32)` | Creates a random neural network model that emphasizes specific features. |
| `CreateRandomVectorModelWithFeatures(Int32[],Int32)` | Creates a random vector model that emphasizes specific features. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_numOps` | Numeric operations provider for type T. |

