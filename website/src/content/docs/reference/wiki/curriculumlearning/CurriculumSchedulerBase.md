---
title: "CurriculumSchedulerBase<T>"
description: "Abstract base class for curriculum schedulers providing common functionality."
section: "API Reference"
---

`Base Classes` · `AiDotNet.CurriculumLearning.Schedulers`

Abstract base class for curriculum schedulers providing common functionality.

## For Beginners

A curriculum scheduler determines how the training curriculum
progresses over time. It controls when and how many samples are included in training
as the model advances through the curriculum.

## How It Works

**Core Responsibilities:**

**Common Scheduler Types:**

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CurriculumSchedulerBase(Int32,,,Nullable<Int32>)` | Initializes a new instance of the `CurriculumSchedulerBase` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `CurrentEpoch` | Gets the current epoch number. |
| `CurrentPhase` | Gets the current phase as a value between 0 and 1. |
| `CurrentPhaseNumber` | Gets the current phase number (0-indexed). |
| `IsComplete` | Gets whether the curriculum has completed (all samples available). |
| `MaxFraction` | Maximum data fraction to end with (usually 1.0). |
| `MinFraction` | Minimum data fraction to start with. |
| `Name` | Gets the name of this scheduler. |
| `TotalEpochs` | Total number of epochs for training. |
| `TotalPhases` | Gets the total number of phases in the curriculum. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AdvancePhase` | Advances to the next phase. |
| `GetCurrentIndices(Int32[],Int32)` | Gets the indices of samples available at the current phase. |
| `GetDataFraction` | Gets the data fraction available at the current phase. |
| `GetDifficultyThreshold` | Gets the difficulty threshold for the current phase. |
| `GetIndicesAtPhase(Int32[],Int32,)` | Gets the indices of samples available at a specific phase. |
| `GetStatistics` | Gets scheduler-specific statistics. |
| `InterpolateFraction()` | Interpolates between min and max fraction based on progress. |
| `Reset` | Resets the scheduler to the initial phase. |
| `StepEpoch(CurriculumEpochMetrics<>)` | Updates the scheduler after an epoch. |

## Fields

| Field | Summary |
|:-----|:--------|
| `NumOps` | Numeric operations for type T. |
| `_currentPhaseNumber` | Current phase number (0-indexed). |
| `_totalPhases` | Number of phases in the curriculum (for phase-based schedulers). |

