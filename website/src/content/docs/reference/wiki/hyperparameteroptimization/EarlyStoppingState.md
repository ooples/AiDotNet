---
title: "EarlyStoppingState"
description: "Represents the current state of early stopping."
section: "API Reference"
---

`Models & Types` · `AiDotNet.HyperparameterOptimization`

Represents the current state of early stopping.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `EarlyStoppingState(Boolean,Double,Int32,Int32,Int32,Int32)` | Initializes a new EarlyStoppingState. |

## Properties

| Property | Summary |
|:-----|:--------|
| `BestEpoch` | The epoch at which the best value was observed. |
| `BestValue` | The best value observed. |
| `EpochsSinceBest` | Number of epochs since the best value was observed. |
| `Patience` | The patience value configured. |
| `Stopped` | Whether early stopping has been triggered. |
| `TotalChecks` | Total number of checks performed. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ToString` |  |

