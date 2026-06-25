---
title: "ExpertDefinedDifficultyEstimator<T, TInput, TOutput>"
description: "Difficulty estimator using pre-defined or expert-provided difficulty scores."
section: "API Reference"
---

`Models & Types` · `AiDotNet.CurriculumLearning.DifficultyEstimators`

Difficulty estimator using pre-defined or expert-provided difficulty scores.

## For Beginners

This estimator uses difficulty scores provided by domain
experts or predefined heuristics. Unlike model-based estimators, it doesn't require
a trained model to estimate difficulty.

## How It Works

**Use Cases:**

**Difficulty Sources:**

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ExpertDefinedDifficultyEstimator(Func<,,>,Boolean)` | Initializes a new instance with a difficulty function. |
| `ExpertDefinedDifficultyEstimator(Vector<>,Boolean)` | Initializes a new instance with precomputed difficulty scores. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Name` | Gets the name of this estimator. |
| `RequiresModel` | Gets whether this estimator requires the model. |

## Methods

| Method | Summary |
|:-----|:--------|
| `EstimateDifficulties(IDataset<,,>,IFullModel<,,>)` | Estimates difficulty scores for all samples in a dataset. |
| `EstimateDifficulty(,,IFullModel<,,>)` | Estimates the difficulty of a single sample. |
| `FromIndex(Int32,Boolean)` | Creates an estimator with difficulty based on sample index. |
| `Random(Int32,Nullable<Int32>)` | Creates an estimator with random difficulty scores. |
| `Update(Int32,IFullModel<,,>)` | Updates the difficulty estimator (no-op for expert-defined). |

