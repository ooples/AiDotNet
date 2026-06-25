---
title: "MMDDomainAdapter<T>"
description: "Implements domain adaptation using Maximum Mean Discrepancy (MMD)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.TransferLearning.DomainAdaptation`

Implements domain adaptation using Maximum Mean Discrepancy (MMD).

## For Beginners

Maximum Mean Discrepancy (MMD) is a way to measure how different two
distributions are. Think of it like comparing the "average characteristics" of two groups.
This adapter minimizes the difference between the average properties of source and target data.

## How It Works

Imagine you have photos from two different cameras. MMD would measure how different the
"average photo" from each camera is, and then adjust them to have similar average properties.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MMDDomainAdapter(Double)` | Initializes a new instance of the MMDDomainAdapter class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AdaptationMethod` | Gets the name of the adaptation method. |
| `RequiresTraining` | Gets whether this adapter requires training. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AdaptSource(Matrix<>,Matrix<>)` | Adapts source data to match target distribution using MMD. |
| `AdaptTarget(Matrix<>,Matrix<>)` | Adapts target data to match source distribution. |
| `ApplyShift(Matrix<>,Vector<>)` | Applies a shift to data. |
| `ComputeDistributionShift(Matrix<>,Matrix<>)` | Computes the distribution shift between two domains. |
| `ComputeDomainDiscrepancy(Matrix<>,Matrix<>)` | Computes the Maximum Mean Discrepancy between two domains. |
| `ComputeKernelSum(Matrix<>,Matrix<>)` | Computes the sum of kernel evaluations between two datasets. |
| `ComputeMeanEmbedding(Matrix<>)` | Computes the mean embedding of data in kernel space. |
| `ComputeMedianHeuristic(Matrix<>,Matrix<>)` | Computes the median heuristic for kernel bandwidth selection. |
| `Train(Matrix<>,Matrix<>)` | Trains the adapter (no-op for MMD as it's non-parametric). |

