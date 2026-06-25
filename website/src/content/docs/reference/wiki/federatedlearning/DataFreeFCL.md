---
title: "DataFreeFCL<T>"
description: "Implements Data-Free Federated Continual Learning — prevents forgetting without storing real data."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.ContinualLearning`

Implements Data-Free Federated Continual Learning — prevents forgetting without storing real data.

## For Beginners

Most continual learning methods require storing old training data
(replay buffers) to remember previous tasks. This is a problem in federated learning because
clients may not be allowed to store data (privacy constraints, storage limits). Data-Free FCL
instead uses the global model itself to generate synthetic "pseudo-samples" that capture
knowledge of previous tasks. These synthetic samples are used during training on new tasks
to prevent forgetting, without ever storing or sharing real client data.

## How It Works

Algorithm:

Reference: Data-Free Federated Continual Learning (2024). Extends
Luo et al., "Data-Free Knowledge Distillation for Heterogeneous FL," NeurIPS 2023.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DataFreeFCL(Double,Double,Int32,Int32)` | Creates a new Data-Free FCL strategy. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DistillationTemperature` | Gets the distillation temperature. |
| `DistillationWeight` | Gets the distillation weight. |
| `GenerationSteps` | Gets the number of generation optimization steps. |
| `SyntheticSamplesPerClass` | Gets the number of synthetic samples per class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AggregateImportance(Dictionary<Int32,Vector<>>,Dictionary<Int32,Double>)` |  |
| `ComputeDistillationLoss(Double[],Double[])` | Computes the knowledge distillation loss between teacher and student soft predictions. |
| `ComputeImportance(Vector<>,Matrix<>)` |  |
| `ComputeRegularizationPenalty(Vector<>,Vector<>,Vector<>,Double)` |  |
| `GenerateSyntheticData(Func<Double[],Double[]>,Int32,Int32,Int32)` | Generates synthetic pseudo-samples that activate the teacher model's knowledge for a target class. |
| `ProjectGradient(Vector<>,Vector<>)` |  |

