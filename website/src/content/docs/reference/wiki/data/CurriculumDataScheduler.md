---
title: "CurriculumDataScheduler"
description: "Schedules training data presentation order and pacing for curriculum learning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Quality`

Schedules training data presentation order and pacing for curriculum learning.

## How It Works

Curriculum learning presents training samples in a meaningful order (easy to hard).
This scheduler determines which samples are available at each epoch based on
difficulty scores and a pacing function that controls data pool growth.

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeFraction(Int32)` | Computes the fraction of data available at a given epoch based on the pacing function. |
| `GetAvailableIndices(Double[],Int32)` | Gets the indices available for training at a given epoch. |

