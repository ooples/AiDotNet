---
title: "VectorHelper"
description: "Provides helper methods for creating and manipulating vectors used in AI and machine learning operations."
section: "API Reference"
---

`Helpers & Utilities` · `AiDotNet.Helpers`

Provides helper methods for creating and manipulating vectors used in AI and machine learning operations.

## How It Works

**For Beginners:** In AI and machine learning, a vector is simply a list of numbers arranged in a specific order.
Think of it as a one-dimensional array or a single column/row of data. Vectors are used to represent:

- Features of a single data point (like height, weight, age of a person)
- Target values we want to predict
- Weights in a trained model
- Intermediate calculations during model training

This helper class provides convenient methods to work with vectors in your AI applications.

## Methods

| Method | Summary |
|:-----|:--------|
| `CosineSimilarity(Vector<>,Vector<>,Double)` | Computes the cosine similarity between two vectors, returning a value in [-1, 1]. |
| `CreateVector(Int32)` | Creates a new vector with the specified size. |
| `DotProduct(Vector<>,Vector<>)` | Computes the dot product of two vectors using hardware-accelerated operations. |
| `EuclideanDistance(Vector<>,Vector<>)` | Computes the Euclidean distance between two vectors using hardware-accelerated operations. |
| `L2Norm(Vector<>)` | Computes the L2 (Euclidean) norm of a vector using hardware-accelerated operations. |
| `ManhattanDistance(Vector<>,Vector<>)` | Computes the Manhattan (L1) distance between two vectors using hardware-accelerated operations. |
| `Normalize(Vector<>,Double)` | Returns a new unit-length vector in the same direction as the input. |
| `NormalizeInPlace(Vector<>,Double)` | Normalizes a vector in place, modifying the original vector to have unit length. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_cosineEngine` | Stable CPU engine used for similarity reductions so the result does not depend on whichever engine is globally active. |

