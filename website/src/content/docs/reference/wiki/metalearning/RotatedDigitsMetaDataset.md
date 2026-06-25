---
title: "RotatedDigitsMetaDataset<T, TInput, TOutput>"
description: "A synthetic meta-dataset for image-like classification where each class is a rotated \"digit\" pattern (a simple 2D feature vector derived from an angle)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.MetaLearning.Data`

A synthetic meta-dataset for image-like classification where each class is a rotated "digit"
pattern (a simple 2D feature vector derived from an angle). Different tasks use different
subsets of rotation angles, simulating few-shot image classification benchmarks.

## For Beginners

This creates a simplified version of the Omniglot-style benchmark.
Each "digit" is represented by a rotation angle that produces a 2D feature vector via
(cos(angle), sin(angle)) plus noise. The meta-learner must learn to classify which rotation
group a new point belongs to from just a few examples.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `RotatedDigitsMetaDataset(Int32,Int32,Int32,Double,Nullable<Int32>)` | Creates a rotated digits meta-dataset. |

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
| `GenerateExample(Matrix<>,Int32,Double)` | Generates a single feature vector for the given base angle with harmonic extensions and noise. |
| `SampleTaskCore(Int32,Int32,Int32)` |  |

