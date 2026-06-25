---
title: "IDifficultyEstimator<T, TInput, TOutput>"
description: "Interface for estimating the difficulty of training samples."
section: "API Reference"
---

`Interfaces` · `AiDotNet.CurriculumLearning.Interfaces`

Interface for estimating the difficulty of training samples.

## For Beginners

A difficulty estimator measures how "hard" each training
sample is for the model to learn. This is crucial for curriculum learning, as we
want to present easy samples first and gradually introduce harder ones.

## How It Works

**Common Difficulty Measures:**

**Research Background:**

## Properties

| Property | Summary |
|:-----|:--------|
| `Name` | Gets the name of the difficulty estimator. |
| `RequiresModel` | Gets whether this estimator requires the model to estimate difficulty. |

## Methods

| Method | Summary |
|:-----|:--------|
| `EstimateDifficulties(IDataset<,,>,IFullModel<,,>)` | Estimates difficulty scores for all samples in a dataset. |
| `EstimateDifficulty(,,IFullModel<,,>)` | Estimates the difficulty of a single sample. |
| `GetSortedIndices(Vector<>)` | Gets the indices of samples sorted by difficulty (easy to hard). |
| `Reset` | Resets the estimator to its initial state. |
| `Update(Int32,IFullModel<,,>)` | Updates the difficulty estimator based on training progress. |

