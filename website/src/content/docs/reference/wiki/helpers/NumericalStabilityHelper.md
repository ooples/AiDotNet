---
title: "NumericalStabilityHelper"
description: "Provides numerical stability utilities for safe mathematical operations in machine learning."
section: "API Reference"
---

`Helpers & Utilities` · `AiDotNet.Helpers`

Provides numerical stability utilities for safe mathematical operations in machine learning.

## For Beginners

Machine learning algorithms often deal with very small or very large numbers,
which can cause numerical issues like:

- Division by zero
- Log of zero or negative numbers
- NaN (Not a Number) values appearing in calculations
- Infinity values from overflow

This helper provides safe versions of common operations that avoid these problems.

## Methods

| Method | Summary |
|:-----|:--------|
| `AssertFinite(,String)` | Asserts that a value is finite, throwing if not. |
| `AssertFinite(Tensor<>,String)` | Asserts that a tensor contains only finite values. |
| `AssertFinite(Vector<>,String)` | Asserts that a vector contains only finite values. |
| `ClampProbability(,Double)` | Clamps a value to valid probability range [epsilon, 1-epsilon]. |
| `ContainsInfinity(Tensor<>)` | Checks if a tensor contains any infinite values. |
| `ContainsInfinity(Vector<>)` | Checks if a vector contains any infinite values. |
| `ContainsNaN(Tensor<>)` | Checks if a tensor contains any NaN values. |
| `ContainsNaN(Vector<>)` | Checks if a vector contains any NaN values. |
| `ContainsNonFinite(Tensor<>)` | Checks if a tensor contains any non-finite values (NaN or infinite). |
| `ContainsNonFinite(Vector<>)` | Checks if a vector contains any non-finite values (NaN or infinite). |
| `CountInfinity(Vector<>)` | Counts the number of infinite values in a vector. |
| `CountNaN(Vector<>)` | Counts the number of NaN values in a vector. |
| `GetEpsilon(Nullable<Double>)` | Gets a type-appropriate epsilon value for the numeric type T. |
| `IsFinite()` | Checks if a value is finite (not NaN and not infinite). |
| `IsInfinity()` | Checks if a value is infinite (positive or negative infinity). |
| `IsNaN()` | Checks if a value is NaN (Not a Number). |
| `ReplaceInfinity(Vector<>,)` | Replaces infinite values in a vector with a specified replacement value. |
| `ReplaceNaN(Vector<>,)` | Replaces NaN values in a vector with a specified replacement value. |
| `ReplaceNonFinite(Vector<>,)` | Replaces all non-finite values (NaN and infinity) in a vector. |
| `SafeDiv(,,Double)` | Performs safe division, avoiding division by zero. |
| `SafeLog(,Double)` | Computes the natural logarithm safely, avoiding log(0) and log(negative). |
| `SafeLogProbability(,Double)` | Computes safe log of a probability (clamps first, then takes log). |
| `SafeSqrt(,Double)` | Computes square root safely, ensuring non-negative input. |
| `StableLogSoftmax(Vector<>)` | Computes log-softmax with numerical stability. |
| `StableSoftmax(Vector<>)` | Computes softmax with numerical stability using the log-sum-exp trick. |

## Fields

| Field | Summary |
|:-----|:--------|
| `DefaultEpsilon` | Default epsilon value for numerical stability (1e-7 for float precision). |
| `LargeEpsilon` | Larger epsilon for less sensitive operations (1e-5). |
| `SmallEpsilon` | Smaller epsilon for double precision operations (1e-15). |

