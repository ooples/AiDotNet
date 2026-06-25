---
title: "ElasticWeightConsolidation<T>"
description: "Implements Elastic Weight Consolidation (EWC) for continual learning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ContinualLearning`

Implements Elastic Weight Consolidation (EWC) for continual learning.

## For Beginners

Elastic Weight Consolidation is like putting rubber bands on
important parts of a neural network. When the network learns a new task, these rubber bands
pull the weights back toward their original values, preventing the network from forgetting
what it learned before.

## How It Works

**How it works:**

**Reference:** Kirkpatrick et al., "Overcoming catastrophic forgetting in neural
networks" (2017). PNAS.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ElasticWeightConsolidation(Double)` | Initializes a new instance of the ElasticWeightConsolidation class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AccumulatesAcrossTasks` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `AfterTask(INeuralNetwork<>,ValueTuple<Tensor<>,Tensor<>>,Int32)` |  |
| `BeforeTask(INeuralNetwork<>,Int32)` |  |
| `ComputeLogLikelihoodGradient(Tensor<>,Tensor<>)` | Computes the gradient of the log-likelihood (negative cross-entropy gradient). |
| `ComputeLoss(INeuralNetwork<>)` |  |
| `ExtractSample(Tensor<>,Int32)` | Extracts a single sample from a batch tensor. |
| `ModifyGradients(INeuralNetwork<>,Vector<>)` |  |
| `Reset` |  |

