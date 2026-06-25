---
title: "ComplexityBasedDifficultyEstimator<T, TInput, TOutput>"
description: "Difficulty estimator based on sample complexity metrics."
section: "API Reference"
---

`Models & Types` · `AiDotNet.CurriculumLearning.DifficultyEstimators`

Difficulty estimator based on sample complexity metrics.

## For Beginners

This estimator calculates difficulty from measurable
properties of the input data, such as magnitude, dimensionality, or variance.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ComplexityBasedDifficultyEstimator(ComplexityMetric,Boolean)` | Initializes a new instance of the `ComplexityBasedDifficultyEstimator` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Name` | Gets the name of this estimator. |
| `RequiresModel` | Gets whether this estimator requires the model. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateEntropyComplexity()` | Calculates complexity based on input entropy. |
| `CalculateMagnitudeComplexity()` | Calculates complexity based on input magnitude. |
| `CalculateSparsityComplexity()` | Calculates complexity based on input sparsity (inverse). |
| `CalculateVarianceComplexity()` | Calculates complexity based on input variance. |
| `EstimateDifficulties(IDataset<,,>,IFullModel<,,>)` | Estimates difficulty scores for all samples. |
| `EstimateDifficulty(,,IFullModel<,,>)` | Estimates difficulty based on input complexity. |

