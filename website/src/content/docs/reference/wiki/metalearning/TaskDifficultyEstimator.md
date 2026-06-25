---
title: "TaskDifficultyEstimator<T>"
description: "Estimates the difficulty of a meta-learning task based on geometric properties of the data: inter-class separation, intra-class variance, and support/query alignment."
section: "API Reference"
---

`Helpers & Utilities` · `AiDotNet.MetaLearning.Data`

Estimates the difficulty of a meta-learning task based on geometric properties of the data:
inter-class separation, intra-class variance, and support/query alignment.

## For Beginners

This utility looks at the support and query data and estimates
how hard the task is. Tasks where classes overlap a lot are harder than tasks where classes
are clearly separated. This estimate can feed into curriculum or dynamic samplers.

## How It Works

**Reference:** Adaptive Task Sampling for Meta-Learning (2024).

## Methods

| Method | Summary |
|:-----|:--------|
| `EstimateDifficulty(Vector<>,Vector<>,Int32,Int32)` | Estimates difficulty of a task in [0, 1] based on support set geometry. |
| `EstimateFisherDifficulty(Vector<>,Vector<>,Int32)` | Estimates difficulty using the Fisher discriminant ratio: difficulty = 1 - (between-class variance / total variance). |

