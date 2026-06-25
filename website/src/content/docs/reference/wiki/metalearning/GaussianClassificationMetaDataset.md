---
title: "GaussianClassificationMetaDataset<T, TInput, TOutput>"
description: "A synthetic meta-dataset for classification where each class is a Gaussian blob in feature space."
section: "API Reference"
---

`Models & Types` · `AiDotNet.MetaLearning.Data`

A synthetic meta-dataset for classification where each class is a Gaussian blob in
feature space. Each task consists of N classes with randomly sampled means and a
shared covariance, making it a standard benchmark for few-shot classification.

## For Beginners

This dataset creates classification tasks where each class is a
cluster of points centered around a random location. The meta-learner must learn to
quickly identify which cluster a new point belongs to, even with very few examples.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GaussianClassificationMetaDataset(Int32,Int32,Int32,Double,Double,Nullable<Int32>)` | Creates a Gaussian classification meta-dataset. |

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
| `SampleGaussian` | Samples from a standard normal distribution using Box-Muller transform. |
| `SampleTaskCore(Int32,Int32,Int32)` |  |

