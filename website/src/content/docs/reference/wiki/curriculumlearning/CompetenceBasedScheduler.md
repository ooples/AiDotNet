---
title: "CompetenceBasedScheduler<T>"
description: "Curriculum scheduler that advances based on model competence/mastery."
section: "API Reference"
---

`Models & Types` · `AiDotNet.CurriculumLearning.Schedulers`

Curriculum scheduler that advances based on model competence/mastery.

## For Beginners

This scheduler tracks how well the model is learning
and only advances to harder content when the model has "mastered" the current
difficulty level. Think of it like a tutor who won't move to harder problems
until you've shown you understand the current ones.

## How It Works

**How It Works:**

**Competence Metrics:**

**References:**

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CompetenceBasedScheduler(Int32,,CompetenceMetricType,Int32,,,,,Nullable<Int32>)` | Initializes a new instance of the `CompetenceBasedScheduler` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `CompetenceThreshold` | Gets or sets the competence threshold required to advance phases. |
| `CurrentCompetence` | Gets the current competence level of the model. |
| `Name` | Gets the name of this scheduler. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeAccuracyCompetence(CurriculumEpochMetrics<>)` | Computes competence based on accuracy metrics. |
| `ComputeCombinedCompetence(CurriculumEpochMetrics<>)` | Computes competence as a combination of multiple metrics. |
| `ComputeLossReductionCompetence(CurriculumEpochMetrics<>)` | Computes competence based on loss reduction. |
| `ComputePlateauCompetence(CurriculumEpochMetrics<>)` | Computes competence based on learning plateau detection. |
| `GetDataFraction` | Gets the current data fraction based on competence-driven phase. |
| `GetStatistics` | Gets scheduler-specific statistics. |
| `HasMasteredCurrentContent` | Gets whether the model has mastered the current curriculum content. |
| `Reset` | Resets the scheduler to initial state. |
| `ResetForNewPhase` | Resets tracking metrics for a new phase. |
| `ResolveDefault(,Double)` | Resolves a nullable generic parameter, returning the fallback if the value is null or default(T). |
| `StepEpoch(CurriculumEpochMetrics<>)` | Updates the scheduler after an epoch, potentially advancing phases. |
| `UpdateCompetence(CurriculumEpochMetrics<>)` | Updates the competence estimate based on model performance. |
| `ValidateThreshold()` | Validates the competence threshold value. |

