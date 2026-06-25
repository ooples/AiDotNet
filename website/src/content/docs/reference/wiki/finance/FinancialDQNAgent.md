---
title: "FinancialDQNAgent<T>"
description: "Financial Deep Q-Network (DQN) agent for discrete action trading."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Finance.Trading.Agents`

Financial Deep Q-Network (DQN) agent for discrete action trading.

## For Beginners

The DQN (Deep Q-Network) trading agent learns to make
discrete trading decisions (buy, sell, or hold) by estimating the long-term value of
each action. It maintains a "memory" of past experiences and learns from random
samples of those memories. A separate target network prevents the learning from
becoming unstable. DQN is best suited for trading scenarios with a fixed set of
possible actions.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FinancialDQNAgent` | Initializes a new instance of the FinancialDQNAgent class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `FeatureCount` |  |
| `ParameterCount` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyGradients(Vector<>,)` |  |
| `Clone` |  |
| `ComputeGradients(Vector<>,Vector<>,ILossFunction<>)` |  |
| `Deserialize(Byte[])` |  |
| `GetActionIndex(Vector<>)` | Executes GetActionIndex for the FinancialDQNAgent. |
| `GetMaxQ(Tensor<>)` | Executes GetMaxQ for the FinancialDQNAgent. |
| `GetModelMetadata` |  |
| `GetOptions` |  |
| `GetParameters` |  |
| `LoadModel(String)` | Executes LoadModel for the FinancialDQNAgent. |
| `SaveModel(String)` | Executes SaveModel for the FinancialDQNAgent. |
| `SelectAction(Vector<>,Boolean)` |  |
| `Serialize` |  |
| `SetParameters(Vector<>)` |  |
| `StoreExperience(Vector<>,Vector<>,,Vector<>,Boolean)` | Executes StoreExperience for the FinancialDQNAgent. |
| `Train` |  |
| `UpdateTargetNetwork` | Executes UpdateTargetNetwork for the FinancialDQNAgent. |

