---
title: "NearMissUnderSampler<T>"
description: "NearMissUnderSampler<T> — Models & Types in AiDotNet.Augmentation.Tabular.Undersampling."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Augmentation.Tabular.Undersampling`

_No summary documentation available yet._

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `NearMissUnderSampler(NearMissUnderSampler<>.NearMissVersion,Int32,Double)` | Creates a new NearMiss undersampler. |

## Properties

| Property | Summary |
|:-----|:--------|
| `NNeighbors` | Gets the number of nearest neighbors to consider. |
| `SamplingRatio` | Gets the target ratio between minority and majority samples after undersampling. |
| `Version` | Gets the NearMiss version to use. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeDistance(Matrix<>,Int32,Int32,Int32)` | Computes Euclidean distance between two samples. |
| `SelectNearMiss1(Matrix<>,List<Int32>,List<Int32>,Int32)` | NearMiss-1: Select majority samples closest to k nearest minority samples. |
| `SelectNearMiss2(Matrix<>,List<Int32>,List<Int32>,Int32)` | NearMiss-2: Select majority samples closest to k farthest minority samples. |
| `SelectNearMiss3(Matrix<>,List<Int32>,List<Int32>,Int32)` | NearMiss-3: For each minority sample, keep k nearest majority samples. |
| `Undersample(Matrix<>,Vector<>,)` | Performs NearMiss undersampling on the majority class. |

