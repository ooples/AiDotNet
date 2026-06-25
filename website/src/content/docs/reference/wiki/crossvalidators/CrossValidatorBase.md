---
title: "CrossValidatorBase<T, TInput, TOutput>"
description: "Provides a base implementation for cross-validation strategies in machine learning models."
section: "API Reference"
---

`Base Classes` · `AiDotNet.CrossValidators`

Provides a base implementation for cross-validation strategies in machine learning models.

## For Beginners

Cross-validation is a technique used to assess how well a machine learning
model will perform on new, unseen data. This base class provides the common structure and
functionality that all cross-validation methods share. Think of it as a blueprint for
creating different types of cross-validation strategies.

## How It Works

This abstract class serves as a foundation for implementing various cross-validation strategies.
It encapsulates common functionality such as numeric operations, random number generation,
and the core cross-validation process. Specific cross-validation types can be implemented
by deriving from this class and providing their own fold creation logic.

Key features:

- Manages numeric operations and random number generation.
- Provides a common method for performing cross-validation once folds are created.
- Allows for easy implementation of various cross-validation strategies by extending this class.
- Supports generic input and output types for flexibility with different data formats.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CrossValidatorBase(CrossValidationOptions)` | Initializes a new instance of the CrossValidationBase class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `PerformCrossValidation(IFullModel<,,>,,,IEnumerable<ValueTuple<Int32[],Int32[]>>,IOptimizer<,,>)` | Executes the cross-validation process using the provided model, data, folds, and optimizer. |
| `Validate(IFullModel<,,>,,,IOptimizer<,,>)` | Performs cross-validation on the given model using the provided data, options, and optimizer. |

## Fields

| Field | Summary |
|:-----|:--------|
| `NumOps` | Provides operations for numeric calculations specific to type T. |
| `Options` | Holds configuration options for cross-validation. |
| `Random` | Random number generator for operations that require randomness. |

