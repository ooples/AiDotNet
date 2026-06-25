---
title: "DifficultyEstimatorBase<T, TInput, TOutput>"
description: "Base class for difficulty estimators."
section: "API Reference"
---

`Base Classes` · `AiDotNet.CurriculumLearning.DifficultyEstimators`

Base class for difficulty estimators.

## For Beginners

This base class provides common functionality for all
difficulty estimators. It handles the mechanics of computing difficulty scores
and sorting samples by difficulty.

## Properties

| Property | Summary |
|:-----|:--------|
| `CacheScores` | Gets or sets whether to cache difficulty scores. |
| `CachedScores` | Gets the cached difficulty scores (if caching is enabled). |
| `HasCachedScores` | Gets whether scores have been computed and cached. |
| `Name` | Gets the name of the difficulty estimator. |
| `RequiresModel` | Gets whether this estimator requires the model to estimate difficulty. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeMean(Vector<>)` | Computes the mean of a vector. |
| `ComputeStandardDeviation(Vector<>,)` | Computes the standard deviation of a vector. |
| `EstimateDifficulties(IDataset<,,>,IFullModel<,,>)` | Estimates difficulty scores for all samples in a dataset. |
| `EstimateDifficulty(,,IFullModel<,,>)` | Estimates the difficulty of a single sample. |
| `GetSortedIndices(Vector<>)` | Gets the indices of samples sorted by difficulty (easy to hard). |
| `NormalizeDifficulties(Vector<>)` | Normalizes difficulty scores to [0, 1] range. |
| `Reset` | Resets the estimator to its initial state. |
| `Update(Int32,IFullModel<,,>)` | Updates the difficulty estimator based on training progress. |

## Fields

| Field | Summary |
|:-----|:--------|
| `NumOps` | Numeric operations for type T. |

