---
title: "FedMaAggregationStrategy<T>"
description: "Implements FedMA (Federated Matched Averaging) aggregation strategy."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.Aggregators`

Implements FedMA (Federated Matched Averaging) aggregation strategy.

## For Beginners

Neural networks have a "permutation problem" — two networks
can compute the same function but have their neurons in different orders. FedMA solves this
by finding the best alignment (matching) of neurons between client models before averaging
them, producing a more accurate global model.

## How It Works

The algorithm:

Reference: Wang, H., et al. (2020). "Federated Learning with Matched Averaging."
ICLR 2020.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FedMaAggregationStrategy(Int32,Double)` | Initializes a new instance of the `FedMaAggregationStrategy` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `MatchingIterations` | Gets the number of matching refinement iterations. |
| `MatchingThreshold` | Gets the cosine similarity threshold for neuron matching. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Aggregate(Dictionary<Int32,Dictionary<String,[]>>,Dictionary<Int32,Double>)` |  |
| `EstimateNeuronCount(Int32)` | Estimates the number of neurons in a flattened weight layer. |
| `GetStrategyName` |  |
| `HungarianAlgorithm(Double[0:,0:],Int32)` | Hungarian algorithm (Kuhn-Munkres) for optimal assignment. |

