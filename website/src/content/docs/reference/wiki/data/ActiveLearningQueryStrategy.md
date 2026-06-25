---
title: "ActiveLearningQueryStrategy"
description: "Selects the most informative unlabeled samples for annotation using active learning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Quality`

Selects the most informative unlabeled samples for annotation using active learning.

## How It Works

Active learning reduces labeling costs by selecting samples that maximize model improvement.
Supports uncertainty sampling, margin sampling, least confidence, and BALD strategies.
Works on model prediction probabilities from the unlabeled pool.

## Methods

| Method | Summary |
|:-----|:--------|
| `Query(Double[][])` | Selects the most informative samples from an unlabeled pool. |
| `QueryBALD(Double[][][])` | Selects samples using BALD (Bayesian Active Learning by Disagreement). |

