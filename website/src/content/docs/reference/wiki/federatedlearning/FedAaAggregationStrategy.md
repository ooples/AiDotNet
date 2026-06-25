---
title: "FedAaAggregationStrategy<T>"
description: "Implements FedAA (Federated Adaptive Aggregation) strategy."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.Aggregators`

Implements FedAA (Federated Adaptive Aggregation) strategy.

## For Beginners

In standard FedAvg, each client's contribution is weighted only
by sample count. FedAA learns better aggregation weights by measuring how similar each
client's update direction is to the overall update direction. Clients whose updates are
more "aligned" with the consensus get higher weight, while outlier updates get lower weight.

## How It Works

Weight computation:

Reference: Adaptive Aggregation for Federated Learning (2024).

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FedAaAggregationStrategy(Double)` | Initializes a new instance of the `FedAaAggregationStrategy` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Temperature` | Gets the softmax temperature for attention weighting. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Aggregate(Dictionary<Int32,Dictionary<String,[]>>,Dictionary<Int32,Double>)` |  |
| `GetStrategyName` |  |

