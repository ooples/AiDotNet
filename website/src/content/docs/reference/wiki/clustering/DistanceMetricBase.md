---
title: "DistanceMetricBase<T>"
description: "Abstract base class for distance metrics providing common functionality."
section: "API Reference"
---

`Base Classes` · `AiDotNet.Clustering.DistanceMetrics`

Abstract base class for distance metrics providing common functionality.

## For Beginners

This is the foundation for all distance metrics.
It provides common code so each specific distance metric only needs to
define how to compute the distance between two points.

## How It Works

This base class provides default implementations for batch distance computations
that can be overridden by derived classes for optimized implementations.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DistanceMetricBase` | Initializes a new instance of the distance metric base class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Engine` | Provides hardware-accelerated tensor/vector operations. |
| `Name` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Abs()` | Computes the absolute value. |
| `Compute(Vector<>,Vector<>)` |  |
| `ComputeInline([],[],Int32)` |  |
| `ComputePairwise(Matrix<>)` |  |
| `ComputePairwise(Matrix<>,Matrix<>)` |  |
| `ComputeToAll(Vector<>,Matrix<>)` |  |
| `GetRow(Matrix<>,Int32)` | Gets a row from a matrix as a vector. |
| `Max(,)` | Returns the maximum of two values. |
| `Pow(,Double)` | Computes the power of a value. |
| `Sqrt()` | Computes the square root of a value. |
| `Sum(Vector<>)` | Computes the sum of elements in a vector. |

## Fields

| Field | Summary |
|:-----|:--------|
| `NumOps` | The numeric operations provider for type T. |

