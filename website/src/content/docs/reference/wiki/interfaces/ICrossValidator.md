---
title: "ICrossValidator<T, TInput, TOutput>"
description: "Defines the contract for cross-validation implementations in machine learning models."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Defines the contract for cross-validation implementations in machine learning models.

## For Beginners

This interface is like a blueprint for creating cross-validation tools.

What it does:

- Defines a standard way to perform cross-validation on any machine learning model
- Ensures that all cross-validation implementations will work the same way, regardless of the specific details
- Works with any data format (matrices, tensors, custom structures) through generic type parameters

It's like setting a standard recipe that all cross-validation methods must follow, ensuring consistency
and ease of use across different types of models and data.

## How It Works

This interface specifies the method signature for performing cross-validation on machine learning models.
Cross-validation is a crucial technique for assessing how the results of a statistical analysis will generalize
to an independent data set. It's particularly important in contexts where the goal is prediction, and one wants
to estimate how accurately a predictive model will perform in practice.

## Methods

| Method | Summary |
|:-----|:--------|
| `Validate(IFullModel<,,>,,,IOptimizer<,,>)` | Performs cross-validation on the given model using the provided data and optimizer. |

