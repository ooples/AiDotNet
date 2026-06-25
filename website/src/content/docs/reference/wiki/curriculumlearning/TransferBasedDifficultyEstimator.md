---
title: "TransferBasedDifficultyEstimator<T, TInput, TOutput>"
description: "Difficulty estimator based on transfer learning from a simpler \"teacher\" model."
section: "API Reference"
---

`Models & Types` · `AiDotNet.CurriculumLearning.DifficultyEstimators`

Difficulty estimator based on transfer learning from a simpler "teacher" model.

## For Beginners

This estimator uses the performance gap between a simple
"teacher" model and the main "student" model to estimate difficulty. Samples that
are easy for the simple model but hard for the main model are considered easier,
while samples that are hard for both are considered more difficult.

## How It Works

**How It Works:**

**Difficulty Calculation:**

**References:**

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TransferBasedDifficultyEstimator(IFullModel<,,>,TransferDifficultyMode,Boolean)` | Initializes a new instance of the `TransferBasedDifficultyEstimator` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Name` | Gets the name of this estimator. |
| `RequiresModel` | Gets whether this estimator requires the main model. |
| `TeacherModel` | Gets the teacher model used for comparison. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateCombinedDifficulty(,,IFullModel<,,>)` | Calculates combined difficulty using multiple metrics. |
| `CalculateConfidenceGapDifficulty(,,IFullModel<,,>)` | Calculates difficulty based on confidence gap. |
| `CalculateLossGapDifficulty(,,IFullModel<,,>)` | Calculates difficulty based on loss gap between teacher and student. |
| `CalculateTeacherLossDifficulty(,)` | Calculates difficulty based only on teacher loss. |
| `EstimateDifficulties(IDataset<,,>,IFullModel<,,>)` | Estimates difficulty scores for all samples. |
| `EstimateDifficulty(,,IFullModel<,,>)` | Estimates the difficulty of a single sample using transfer comparison. |
| `GetModelConfidence(IFullModel<,,>,)` | Gets model confidence for a prediction. |

