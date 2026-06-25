---
title: "IDomainAdapter<T>"
description: "Defines the interface for adapting models to reduce distribution shift between source and target domains."
section: "API Reference"
---

`Interfaces` · `AiDotNet.TransferLearning.DomainAdaptation`

Defines the interface for adapting models to reduce distribution shift between source and target domains.

## For Beginners

Domain adaptation is like helping someone adjust to a new environment.
Even when features are mapped correctly, the source and target domains might have different
statistical properties (like different averages or variability). A domain adapter helps
reduce these differences so a model trained on source data works better on target data.

## How It Works

Think of it like adjusting your eyes when moving from a bright room to a dim room - the
objects are the same, but your perception needs to adapt to the new lighting conditions.

## Properties

| Property | Summary |
|:-----|:--------|
| `AdaptationMethod` | Gets the name of the adaptation method. |
| `RequiresTraining` | Determines if the adapter requires training before use. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AdaptSource(Matrix<>,Matrix<>)` | Adapts source domain data to better match the target domain distribution. |
| `AdaptTarget(Matrix<>,Matrix<>)` | Adapts target domain data to better match the source domain distribution. |
| `ComputeDomainDiscrepancy(Matrix<>,Matrix<>)` | Computes the domain discrepancy (how different the domains are). |
| `Train(Matrix<>,Matrix<>)` | Trains the domain adapter if required. |

