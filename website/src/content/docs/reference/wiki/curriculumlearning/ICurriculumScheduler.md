---
title: "ICurriculumScheduler<T>"
description: "Interface for curriculum schedulers that control training progression."
section: "API Reference"
---

`Interfaces` · `AiDotNet.CurriculumLearning.Interfaces`

Interface for curriculum schedulers that control training progression.

## For Beginners

A curriculum scheduler decides when and how to introduce
harder training samples. Think of it like a teacher who decides when students are
ready to move from addition to multiplication.

## How It Works

**Common Scheduling Strategies:**

**Key Concepts:**

## Properties

| Property | Summary |
|:-----|:--------|
| `CurrentEpoch` | Gets the current epoch number. |
| `CurrentPhase` | Gets the current phase (0 to 1). |
| `CurrentPhaseNumber` | Gets the current phase number (0-indexed). |
| `IsComplete` | Gets whether the curriculum is complete (all samples available). |
| `Name` | Gets the name of the scheduler. |
| `TotalPhases` | Gets the total number of phases in the curriculum. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AdvancePhase` | Advances to the next phase. |
| `GetCurrentIndices(Int32[],Int32)` | Gets the indices of samples available at the current phase. |
| `GetDataFraction` | Gets the data fraction available at the current phase. |
| `GetDifficultyThreshold` | Gets the difficulty threshold for the current phase. |
| `GetIndicesAtPhase(Int32[],Int32,)` | Gets the indices of samples available at a specific phase. |
| `GetStatistics` | Gets statistics about the scheduler's current state. |
| `Reset` | Resets the scheduler to the initial phase. |
| `StepEpoch(CurriculumEpochMetrics<>)` | Updates the scheduler after an epoch. |

