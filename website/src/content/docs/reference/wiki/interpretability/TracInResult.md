---
title: "TracInResult<T>"
description: "Result of TracIn computation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Interpretability.Explainers`

Result of TracIn computation.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TracInResult(Vector<>,,Vector<>,Int32)` | Initializes a new TracIn result. |

## Properties

| Property | Summary |
|:-----|:--------|
| `NumCheckpoints` | Gets the number of checkpoints used. |
| `TestInput` | Gets the test input. |
| `TestLabel` | Gets the test label. |
| `TracInScores` | Gets TracIn scores for each training sample. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetTopInfluential(Int32)` | Gets the most influential training samples according to TracIn. |
| `ToString` | Returns a human-readable summary. |

