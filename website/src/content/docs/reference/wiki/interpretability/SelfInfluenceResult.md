---
title: "SelfInfluenceResult<T>"
description: "Result of self-influence computation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Interpretability.Explainers`

Result of self-influence computation.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SelfInfluenceResult(Vector<>,Matrix<>,Vector<>)` | Initializes a new self-influence result. |

## Properties

| Property | Summary |
|:-----|:--------|
| `SelfInfluences` | Gets the self-influence score for each training sample. |
| `TrainingData` | Gets the training data. |
| `TrainingLabels` | Gets the training labels. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetPotentiallyProblematic(Int32)` | Gets samples most likely to be mislabeled or problematic. |
| `ToString` | Returns a human-readable summary. |

