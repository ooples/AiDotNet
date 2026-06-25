---
title: "EnsembleDifficultyEstimator<T, TInput, TOutput>"
description: "Difficulty estimator that combines multiple estimators for robust difficulty estimation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.CurriculumLearning.DifficultyEstimators`

Difficulty estimator that combines multiple estimators for robust difficulty estimation.

## For Beginners

This estimator combines multiple difficulty estimators
into a single, more robust estimate. Different estimators may capture different
aspects of difficulty, and combining them can provide a more comprehensive view.

## How It Works

**Combination Strategies:**

**Benefits:**

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `EnsembleDifficultyEstimator(EnsembleCombinationMethod,Boolean)` | Initializes a new instance of the `EnsembleDifficultyEstimator` class. |
| `EnsembleDifficultyEstimator(IEnumerable<IDifficultyEstimator<,,>>,IEnumerable<>,EnsembleCombinationMethod,Boolean)` | Initializes a new instance with predefined estimators. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Estimators` | Gets the individual estimators in this ensemble. |
| `Name` | Gets the name of this estimator. |
| `RequiresModel` | Gets whether this estimator requires the model. |
| `Weights` | Gets or sets the weights for each estimator. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AddEstimator(IDifficultyEstimator<,,>,)` | Adds an estimator to the ensemble. |
| `CombineDifficulties([])` | Combines difficulty values from multiple estimators. |
| `CombineGeometricMean([])` | Combines using geometric mean. |
| `CombineHarmonicMean([])` | Combines using harmonic mean. |
| `CombineMaximum([])` | Combines using maximum. |
| `CombineMedian([])` | Combines using median. |
| `CombineMinimum([])` | Combines using minimum. |
| `CombineWeightedAverage([])` | Combines using weighted average. |
| `EstimateDifficulties(IDataset<,,>,IFullModel<,,>)` | Estimates difficulty scores for all samples. |
| `EstimateDifficulty(,,IFullModel<,,>)` | Estimates the difficulty of a single sample using all estimators. |
| `RemoveEstimator(Int32)` | Removes an estimator from the ensemble. |
| `Reset` | Resets all estimators in the ensemble. |
| `Update(Int32,IFullModel<,,>)` | Updates all estimators in the ensemble. |

