---
title: "PersonalizedFederatedLearning<T>"
description: "Implements personalized federated learning where each client maintains some client-specific parameters."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.Personalization`

Implements personalized federated learning where each client maintains some client-specific parameters.

## How It Works

Personalized Federated Learning (PFL) addresses the challenge of heterogeneous data distributions
across clients by allowing each client to maintain personalized model components while still
benefiting from collaborative learning.

**For Beginners:** Personalized FL is like having a shared textbook but personal notes.
Everyone learns from the same core material (global model) but adapts it to their specific
needs (personalized layers).

Key concept:

- Global layers: Shared across all clients, learn common patterns
- Personalized layers: Client-specific, adapt to local data distribution
- Clients train both but only global layers are aggregated

How it works:

1. Model is split into global and personalized parts
2. During local training, both parts are updated
3. Only global parts are sent to server for aggregation
4. Personalized parts stay on the client
5. Client receives updated global parts and keeps personalized parts

For example, in healthcare:

- Hospital A: Urban population, young average age
- Hospital B: Rural population, old average age
- Hospital C: Suburban population, mixed age

Model structure:

- Global layers (shared): General disease detection features
- Personalized layers: Adapt to local demographics

Benefits:

- Better performance on non-IID data
- Each client gets a model optimized for their data
- Preserves privacy (personalized parts never leave client)
- Relatively simple to implement

Common approaches:

1. Layer-wise personalization: Last few layers personalized
2. Feature-wise personalization: Some features personalized
3. Meta-learning: Learn how to adapt quickly to local data
4. Multi-task learning: Treat each client as a separate task

When to use PFL:

- Clients have significantly different data distributions
- Standard FedAvg performance is poor
- Can afford client-side storage for personalized parameters
- Want better local performance even at cost of global performance

Limitations:

- Requires more storage on client (for personalized params)
- May sacrifice some global model quality
- Need to choose which layers to personalize
- Risk of overfitting to local data

Reference:

- Wang, K., et al. (2019). "Federated Evaluation of On-device Personalization." arXiv preprint.
- Fallah, A., et al. (2020). "Personalized Federated Learning with Theoretical Guarantees: A Model-Agnostic Meta-Learning Approach." NeurIPS 2020.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PersonalizedFederatedLearning(Double)` | Initializes a new instance of the `PersonalizedFederatedLearning` class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CombineModels(Dictionary<String,[]>,Dictionary<String,[]>)` | Combines global model update with client's personalized layers. |
| `GetModelSplitStatistics(Dictionary<String,[]>)` | Calculates statistics about the model split. |
| `GetPersonalizationFraction` | Gets the personalization fraction. |
| `GetPersonalizedLayers` | Gets the set of all personalized layer names. |
| `IdentifyPersonalizedLayers(Dictionary<String,[]>,PersonalizedLayerSelectionStrategy,HashSet<String>)` | Identifies which layers should be personalized based on the model structure. |
| `IsLayerPersonalized(String)` | Checks if a specific layer is personalized. |
| `SeparateModel(Dictionary<String,[]>,Dictionary<String,[]>,Dictionary<String,[]>)` | Separates a model into global and personalized components. |

