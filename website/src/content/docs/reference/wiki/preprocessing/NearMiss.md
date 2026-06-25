---
title: "NearMiss<T>"
description: "NearMiss<T> — Models & Types in AiDotNet.Preprocessing.ImbalancedLearning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.ImbalancedLearning`

_No summary documentation available yet._

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `NearMiss(Double,NearMissVersion,Int32,Nullable<Int32>)` | Initializes a new instance of the NearMiss class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Name` | Gets the name of this undersampling strategy. |

## Methods

| Method | Summary |
|:-----|:--------|
| `SelectNearMiss1(Matrix<>,List<Int32>,List<Int32>,Int32)` | NearMiss-1: Keep majority samples closest to their nearest minority neighbors. |
| `SelectNearMiss2(Matrix<>,List<Int32>,List<Int32>,Int32)` | NearMiss-2: Keep majority samples closest to their farthest minority neighbors. |
| `SelectNearMiss3(Matrix<>,List<Int32>,List<Int32>,Int32)` | NearMiss-3: Keep k nearest majority neighbors for each minority sample. |
| `SelectSamplesToKeep(Matrix<>,Vector<>,List<Int32>,List<Int32>,Int32)` | Selects which majority samples to keep using NearMiss selection. |

