---
title: "SineWaveMetaDataset<T, TInput, TOutput>"
description: "A synthetic meta-dataset for regression where each task is a sinusoidal function with random amplitude and phase."
section: "API Reference"
---

`Models & Types` · `AiDotNet.MetaLearning.Data`

A synthetic meta-dataset for regression where each task is a sinusoidal function
with random amplitude and phase. This is the standard benchmark from the MAML paper
(Finn et al., ICML 2017).

## For Beginners

This dataset generates sine wave tasks like y = A * sin(x + phase).
Each "class" is defined by a unique (amplitude, phase) pair. The meta-learner must
learn to quickly fit a sine wave from just a few (x, y) points.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SineWaveMetaDataset(Int32,Int32,Double,Double,Double,Double,Double,Double,Nullable<Int32>)` | Creates a sine wave meta-dataset. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ClassExampleCounts` |  |
| `Name` |  |
| `TotalClasses` |  |
| `TotalExamples` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `SampleTaskCore(Int32,Int32,Int32)` |  |

