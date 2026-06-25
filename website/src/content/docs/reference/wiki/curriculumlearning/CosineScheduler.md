---
title: "CosineScheduler<T>"
description: "Curriculum scheduler with cosine annealing curve."
section: "API Reference"
---

`Models & Types` · `AiDotNet.CurriculumLearning.Schedulers`

Curriculum scheduler with cosine annealing curve.

## For Beginners

This scheduler follows a cosine curve, providing
smooth progression that starts slow, accelerates in the middle, and slows
down again near the end. This can help with convergence stability.

## How It Works

**Progression Pattern:**

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CosineScheduler(Int32,,)` | Initializes a new instance of the `CosineScheduler` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Name` | Gets the name of this scheduler. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetDataFraction` | Gets the current data fraction using cosine annealing. |

