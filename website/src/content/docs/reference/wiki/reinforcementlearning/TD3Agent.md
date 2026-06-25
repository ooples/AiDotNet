---
title: "TD3Agent<T>"
description: "Twin Delayed Deep Deterministic Policy Gradient (TD3) agent for continuous control."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ReinforcementLearning.Agents.TD3`

Twin Delayed Deep Deterministic Policy Gradient (TD3) agent for continuous control.

## For Beginners

TD3 is one of the best algorithms for continuous control tasks (like robot movement).
It's more stable and robust than DDPG.

Key innovations:

- **Twin Critics**: Uses two Q-networks and takes the minimum to avoid overoptimism
- **Delayed Updates**: Waits before updating the policy to let Q-values stabilize
- **Target Smoothing**: Adds noise to target actions to prevent exploitation of errors

Think of it like getting a second opinion before making decisions, and taking time
to verify information before acting on it.

Used by: Robotic control, autonomous systems, continuous optimization

## How It Works

TD3 improves upon DDPG with three key innovations:

1. Twin Q-Networks: Uses two Q-functions to reduce overestimation bias
2. Delayed Policy Updates: Updates policy less frequently than Q-networks
3. Target Policy Smoothing: Adds noise to target actions for robustness

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TD3Agent` | Initializes a new instance with default settings. |

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
| `SaveModel(String)` |  |
| `Serialize` |  |
| `SetParameters(Vector<>)` |  |

