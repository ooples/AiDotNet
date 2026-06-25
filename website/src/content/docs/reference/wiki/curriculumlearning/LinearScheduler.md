---
title: "LinearScheduler<T>"
description: "Curriculum scheduler with linear progression from easy to hard samples."
section: "API Reference"
---

`Models & Types` · `AiDotNet.CurriculumLearning.Schedulers`

Curriculum scheduler with linear progression from easy to hard samples.

## For Beginners

This scheduler increases the data fraction linearly over
training epochs. It starts with easy samples and gradually includes more difficult
samples at a constant rate.

## How It Works

**Progression Pattern:**

**Example:** With minFraction=0.2 and maxFraction=1.0 over 10 epochs:

**Best For:**

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LinearScheduler(Int32,,)` | Initializes a new instance of the `LinearScheduler` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Name` | Gets the name of this scheduler. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetDataFraction` | Gets the current data fraction using linear interpolation. |

