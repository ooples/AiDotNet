---
title: "CQLAgent<T>"
description: "Conservative Q-Learning (CQL) agent for offline reinforcement learning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ReinforcementLearning.Agents.CQL`

Conservative Q-Learning (CQL) agent for offline reinforcement learning.

## For Beginners

Unlike online RL (which tries actions and learns), CQL learns only from recorded data.
This is crucial for domains where exploration is dangerous or expensive.

Key features:

- **Conservative Penalty**: Lowers Q-values for unseen state-action pairs
- **Offline Learning**: No environment interaction needed
- **Safe Policy Improvement**: Guarantees improvement over behavior policy

Example use cases:

- Learning from medical records (can't experiment on patients)
- Autonomous driving from dashcam data
- Robotics from demonstration datasets

## How It Works

CQL is designed for offline RL, learning from fixed datasets without environment interaction.
It prevents overestimation by adding a conservative penalty that pushes down Q-values
for out-of-distribution actions while maintaining accuracy on in-distribution actions.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CQLAgent` | Initializes a new instance with default settings. |

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
| `LoadOfflineData(List<ValueTuple<Vector<>,Vector<>,,Vector<>,Boolean>>)` | Load offline dataset into the replay buffer. |
| `SaveModel(String)` |  |
| `Serialize` |  |
| `SetParameters(Vector<>)` |  |

