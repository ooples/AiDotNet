---
title: "A3CAgent<T>"
description: "Asynchronous Advantage Actor-Critic (A3C) agent for reinforcement learning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ReinforcementLearning.Agents.A3C`

Asynchronous Advantage Actor-Critic (A3C) agent for reinforcement learning.

## For Beginners

A3C is like having multiple students learn simultaneously - each has different
experiences, and they periodically share knowledge with a "master" network.
This parallel learning provides stability and diverse exploration.

Key features:

- **Asynchronous Updates**: Multiple workers update global network independently
- **No Replay Buffer**: On-policy learning with parallel exploration
- **Actor-Critic**: Learns both policy and value function
- **Diverse Exploration**: Each worker explores differently

Famous for: DeepMind's breakthrough (2016), enables CPU-only training

## How It Works

A3C runs multiple agents in parallel, each exploring different strategies.
Workers periodically synchronize with a global network, enabling diverse exploration
without replay buffers.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `A3CAgent` | Initializes a new instance with default settings. |

## Properties

| Property | Summary |
|:-----|:--------|
| `FeatureCount` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyGradients(Vector<>,)` |  |
| `Clone` |  |
| `ComputeGradients(Vector<>,Vector<>,ILossFunction<>)` |  |
| `Deserialize(Byte[])` |  |
| `GetModelMetadata` |  |
| `GetOptions` |  |
| `GetParameters` |  |
| `LoadModel(String)` |  |
| `Predict(Vector<>)` | IFullModel.Predict surfaces the raw policy-network output (softmax over discrete actions, or [mean \| logStd] for continuous actions) rather than the one-hot committed action. |
| `SaveModel(String)` |  |
| `Serialize` |  |
| `SetParameters(Vector<>)` |  |
| `StoreExperience(Vector<>,Vector<>,,Vector<>,Boolean)` | Appends one step of (s, a, r, s', done) experience to the on-policy trajectory buffer. |
| `Train` | One A3C update step (Mnih et al. |
| `TrainAsync` | Async wrapper around the sync `Train` step. |
| `TrainAsync(IEnvironment<>,Int32)` | Train A3C with parallel workers (simplified for single-threaded execution). |

## Fields

| Field | Summary |
|:-----|:--------|
| `_trajectory` | On-policy trajectory buffer. |

