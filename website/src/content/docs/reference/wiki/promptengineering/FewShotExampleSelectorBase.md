---
title: "FewShotExampleSelectorBase<T>"
description: "Base class for few-shot example selector implementations."
section: "API Reference"
---

`Base Classes` · `AiDotNet.PromptEngineering.FewShot`

Base class for few-shot example selector implementations.

## For Beginners

This is the foundation for all example selectors.

It handles:

- Storing examples
- Adding/removing examples
- Basic validation

Derived classes just need to implement how to SELECT examples!

## How It Works

This base class provides common functionality for example selectors including example storage
and basic validation. Derived classes implement the selection strategy.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FewShotExampleSelectorBase` | Initializes a new instance of the FewShotExampleSelectorBase class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ExampleCount` | Gets the total number of examples in the pool. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AddExample(FewShotExample)` | Adds an example to the selector's pool. |
| `ClampToUnitInterval()` | Clamps a value to the [0, 1] interval. |
| `CompareDescending(,)` | Compares two values in descending order using `INumericOperations` comparisons. |
| `CosineSimilarity(Vector<>,Vector<>)` | Calculates cosine similarity between two vectors. |
| `EuclideanDistanceSquared(Vector<>,Vector<>)` | Calculates squared Euclidean distance between two vectors. |
| `GetAllExamples` | Gets all examples currently in the selector's pool. |
| `OnExampleAdded(FewShotExample)` | Called when an example is added. |
| `OnExampleRemoved(FewShotExample)` | Called when an example is removed. |
| `RemoveExample(FewShotExample)` | Removes an example from the selector's pool. |
| `SelectExamples(String,Int32)` | Selects the most appropriate examples for the given query. |
| `SelectExamplesCore(String,Int32)` | Core selection logic to be implemented by derived classes. |

## Fields

| Field | Summary |
|:-----|:--------|
| `Examples` | The pool of available examples. |
| `NumOps` | Numeric operations for type `T`. |

