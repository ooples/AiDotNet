---
title: "FedAvgAggregationStrategy<T>"
description: "Implements the Federated Averaging (FedAvg) aggregation strategy."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.Aggregators`

Implements the Federated Averaging (FedAvg) aggregation strategy.

## How It Works

FedAvg is the foundational aggregation algorithm for federated learning, proposed by
McMahan et al. in 2017. It performs a weighted average of client model updates based
on the number of training samples each client has.

**For Beginners:** FedAvg is like calculating a weighted class average where students
who solved more practice problems have more influence on the final answer.

How FedAvg works:

1. Each client trains on their local data and computes model updates
2. Clients send their updated model weights to the server
3. Server computes weighted average: weight = (client_samples / total_samples)
4. New global model = Σ(weight_i × client_model_i)

For example, with 3 hospitals:

- Hospital A: 1000 patients, model accuracy 90%
- Hospital B: 500 patients, model accuracy 88%
- Hospital C: 1500 patients, model accuracy 92%

Total patients: 3000
Hospital A weight: 1000/3000 = 0.333
Hospital B weight: 500/3000 = 0.167
Hospital C weight: 1500/3000 = 0.500

For each model parameter:
global_param = 0.333 × A_param + 0.167 × B_param + 0.500 × C_param

Benefits:

- Simple and efficient
- Well-studied theoretically
- Works well when clients have similar data distributions (IID data)

Limitations:

- Assumes clients are equally reliable
- Can struggle with non-IID data (different distributions across clients)
- No built-in handling for stragglers (slow clients)

Reference: McMahan, H. B., et al. (2017). "Communication-Efficient Learning of Deep Networks
from Decentralized Data." AISTATS 2017.

## Methods

| Method | Summary |
|:-----|:--------|
| `Aggregate(Dictionary<Int32,Dictionary<String,[]>>,Dictionary<Int32,Double>)` | Aggregates client models using weighted averaging based on the number of samples. |
| `GetStrategyName` | Gets the name of the aggregation strategy. |

