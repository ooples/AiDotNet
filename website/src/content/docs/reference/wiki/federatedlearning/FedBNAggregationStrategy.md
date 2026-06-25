---
title: "FedBNAggregationStrategy<T>"
description: "Implements the Federated Batch Normalization (FedBN) aggregation strategy."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.Aggregators`

Implements the Federated Batch Normalization (FedBN) aggregation strategy.

## How It Works

FedBN is a specialized aggregation strategy that handles batch normalization layers
differently from other layers in neural networks. Proposed by Li et al. in 2021,
it addresses the challenge of non-IID data by keeping batch normalization parameters local.

**For Beginners:** FedBN recognizes that some parts of a neural network should remain
personalized to each client rather than being averaged globally.

The key insight:

- Batch Normalization (BN) layers learn statistics specific to each client's data
- Averaging BN parameters across clients with different data distributions hurts performance
- Solution: Keep BN layers local, only aggregate other layers (Conv, FC, etc.)

How FedBN works:

1. During aggregation, identify batch normalization layers
2. Aggregate only non-BN layers using weighted averaging
3. Keep each client's BN layers unchanged (personalized)
4. Send back global model with client-specific BN layers

For example, in a CNN with layers:

- Conv1 (filters) → BN1 (normalization) → ReLU → Conv2 → BN2 → FC (classification)

FedBN aggregates:

- ✓ Conv1 filters: Averaged across clients
- ✗ BN1 params: Kept local to each client
- ✓ Conv2 filters: Averaged across clients
- ✗ BN2 params: Kept local to each client
- ✓ FC weights: Averaged across clients

Why this matters:

- Different clients may have different data ranges, distributions
- Hospital A images: brightness range [0, 100]
- Hospital B images: brightness range [50, 200]
- Each needs different normalization parameters
- Shared feature extractors (Conv layers) + personalized normalization works better

When to use FedBN:

- Training deep neural networks (especially CNNs)
- Non-IID data with distribution shift
- Batch normalization or layer normalization in architecture
- Want to improve accuracy without changing training much

Benefits:

- Significantly improves accuracy on non-IID data
- Simple modification to FedAvg
- No additional communication cost
- Each client keeps personalized normalization

Limitations:

- Only helps when using batch normalization
- Doesn't address other heterogeneity challenges
- Requires identifying BN layers in model structure

Reference: Li, X., et al. (2021). "Federated Learning on Non-IID Data Silos: An Experimental Study."
ICDE 2021.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FedBNAggregationStrategy(HashSet<String>)` | Initializes a new instance of the `FedBNAggregationStrategy` class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Aggregate(Dictionary<Int32,Dictionary<String,[]>>,Dictionary<Int32,Double>)` | Aggregates client models while keeping batch normalization layers local. |
| `GetBatchNormPatterns` | Gets the batch normalization layer patterns used for identification. |
| `GetStrategyName` | Gets the name of the aggregation strategy. |
| `IsBatchNormalizationLayer(String)` | Determines whether a layer is a batch normalization layer based on its name. |

