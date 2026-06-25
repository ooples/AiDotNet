---
title: "StepScheduler<T>"
description: "Curriculum scheduler with discrete step-based progression."
section: "API Reference"
---

`Models & Types` · `AiDotNet.CurriculumLearning.Schedulers`

Curriculum scheduler with discrete step-based progression.

## For Beginners

This scheduler divides training into discrete phases,
with the data fraction jumping at specific epochs rather than changing continuously.

## How It Works

**Example:** With 3 steps over 12 epochs:

**Best For:**

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `StepScheduler(Int32,IEnumerable<>)` | Initializes a new instance with custom step fractions. |
| `StepScheduler(Int32,Int32,,)` | Initializes a new instance with uniform steps. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Name` | Gets the name of this scheduler. |
| `TotalPhases` | Gets the total number of phases (steps) in this scheduler. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetDataFraction` | Gets the current data fraction based on the current step. |
| `GetStatistics` | Gets scheduler-specific statistics. |

