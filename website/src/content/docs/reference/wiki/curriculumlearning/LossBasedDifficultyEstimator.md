---
title: "LossBasedDifficultyEstimator<T, TInput, TOutput>"
description: "Difficulty estimator based on training loss."
section: "API Reference"
---

`Models & Types` · `AiDotNet.CurriculumLearning.DifficultyEstimators`

Difficulty estimator based on training loss.

## For Beginners

This estimator uses the model's prediction loss as a
measure of difficulty. Samples with high loss are considered harder because the
model struggles to predict them correctly.

## How It Works

**How It Works:**

**References:**

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LossBasedDifficultyEstimator(ILossFunction<>,Boolean)` | Initializes a new instance of the `LossBasedDifficultyEstimator` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Name` | Gets the name of this estimator. |
| `RequiresModel` | Gets whether this estimator requires the model. |

## Methods

| Method | Summary |
|:-----|:--------|
| `EstimateDifficulties(IDataset<,,>,IFullModel<,,>)` | Estimates difficulty scores for all samples (batch optimized). |
| `EstimateDifficulty(,,IFullModel<,,>)` | Estimates the difficulty of a single sample based on prediction loss. |

