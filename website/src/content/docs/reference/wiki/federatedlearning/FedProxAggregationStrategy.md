---
title: "FedProxAggregationStrategy<T>"
description: "Implements the Federated Proximal (FedProx) aggregation strategy."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.Aggregators`

Implements the Federated Proximal (FedProx) aggregation strategy.

## How It Works

FedProx is an extension of FedAvg that handles system and statistical heterogeneity
in federated learning. It was proposed by Li et al. in 2020 to address challenges
when clients have different computational capabilities or data distributions.

**For Beginners:** FedProx is like FedAvg with a "safety rope" that prevents
individual clients from pulling the shared model too far in their own direction.

Key differences from FedAvg:

1. Adds a proximal term to local training objective
2. Prevents client models from deviating too much from global model
3. Improves convergence when clients have heterogeneous data or capabilities

How FedProx works:
During local training, each client minimizes:
Local Loss + (μ/2) × ||w - w_global||²

where:

- Local Loss: Standard loss on client's data
- μ (mu): Proximal term coefficient (controls constraint strength)
- w: Client's current model weights
- w_global: Global model weights received from server
- ||w - w_global||²: Squared distance between client and global model

For example, with μ = 0.01:

- Client trains on local data
- Proximal term penalizes large deviations from global model
- If client's data is very different, can still adapt but with limitation
- Prevents overfitting to local data distribution

When to use FedProx:

- Non-IID data (different distributions across clients)
- System heterogeneity (some clients much slower/faster)
- Want more stable convergence than FedAvg
- Stragglers problem (some clients take much longer)

Benefits:

- Better convergence on non-IID data
- More robust to stragglers
- Theoretically proven convergence guarantees
- Small computational overhead

Limitations:

- Requires tuning μ parameter
- Slightly slower local training per iteration
- May converge slower if μ is too large

Reference: Li, T., et al. (2020). "Federated Optimization in Heterogeneous Networks."
MLSys 2020.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FedProxAggregationStrategy(Double)` | Initializes a new instance of the `FedProxAggregationStrategy` class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Aggregate(Dictionary<Int32,Dictionary<String,[]>>,Dictionary<Int32,Double>)` | Aggregates client models using FedProx weighted averaging. |
| `GetMu` | Gets the proximal term coefficient μ. |
| `GetStrategyName` | Gets the name of the aggregation strategy. |

