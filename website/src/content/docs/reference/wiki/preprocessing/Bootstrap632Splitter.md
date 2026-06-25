---
title: "Bootstrap632Splitter<T>"
description: ".632 Bootstrap that provides bias-corrected error estimation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.DataPreparation.Splitting.Bootstrap`

.632 Bootstrap that provides bias-corrected error estimation.

## For Beginners

Regular bootstrap error estimates can be biased (too optimistic).
The .632 bootstrap corrects this by combining training error and OOB error.

## How It Works

**The Formula:**
Error_632 = 0.368 × TrainingError + 0.632 × OOB_Error

**Why .632?**
This number comes from the probability that a sample is NOT in a bootstrap sample:
(1 - 1/n)^n ≈ 1/e ≈ 0.368
So approximately 63.2% of unique samples end up in training.

**Note:** This splitter generates the same splits as regular bootstrap.
The .632 error calculation should be done during model evaluation.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `Bootstrap632Splitter(Int32,Int32)` | Creates a new .632 bootstrap splitter. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Description` |  |

