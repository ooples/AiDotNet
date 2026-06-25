---
title: "ExponentialScheduler<T>"
description: "Curriculum scheduler with exponential (slow start) progression."
section: "API Reference"
---

`Models & Types` · `AiDotNet.CurriculumLearning.Schedulers`

Curriculum scheduler with exponential (slow start) progression.

## For Beginners

This scheduler starts slowly with easy samples and
accelerates the addition of harder samples later in training. It follows an
exponential curve that reaches the maximum fraction at the end.

## How It Works

**Progression Pattern:**

**Growth Rate Parameter:**

**Best For:**

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ExponentialScheduler(Int32,Double,,)` | Initializes a new instance of the `ExponentialScheduler` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Name` | Gets the name of this scheduler. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetDataFraction` | Gets the current data fraction using exponential curve. |
| `GetStatistics` | Gets scheduler-specific statistics. |

