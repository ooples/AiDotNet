---
title: "FinancialSACAgent<T>"
description: "Financial Soft Actor-Critic (SAC) agent for high-performance continuous trading."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Finance.Trading.Agents`

Financial Soft Actor-Critic (SAC) agent for high-performance continuous trading.

## For Beginners

The SAC (Soft Actor-Critic) trading agent is designed for
continuous trading decisions, like choosing exact position sizes (e.g., buy 37% of
portfolio capacity). It encourages exploration by maximizing both returns and the
"entropy" (randomness) of its strategy, which prevents it from getting stuck in a
suboptimal trading pattern. SAC is considered state-of-the-art for continuous action
spaces and adapts well to changing market conditions.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FinancialSACAgent(NeuralNetworkArchitecture<>,NeuralNetworkArchitecture<>,TradingAgentOptions<>)` | Initializes a new instance of the FinancialSACAgent class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `FeatureCount` |  |
| `ParameterCount` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyGradients(Vector<>,)` | Executes ApplyGradients for the FinancialSACAgent. |
| `Clone` | Executes Clone for the FinancialSACAgent. |
| `ComputeGradients(Vector<>,Vector<>,ILossFunction<>)` | Executes ComputeGradients for the FinancialSACAgent. |
| `Deserialize(Byte[])` | Executes Deserialize for the FinancialSACAgent. |
| `GetModelMetadata` | Executes GetModelMetadata for the FinancialSACAgent. |
| `GetOptions` |  |
| `GetParameters` | Gets all trainable parameters from the actor and critic networks. |
| `LoadModel(String)` | Executes LoadModel for the FinancialSACAgent. |
| `SaveModel(String)` | Executes SaveModel for the FinancialSACAgent. |
| `SelectAction(Vector<>,Boolean)` |  |
| `Serialize` | Executes Serialize for the FinancialSACAgent. |
| `SetParameters(Vector<>)` | Sets all trainable parameters for the actor and critic networks. |
| `StoreExperience(Vector<>,Vector<>,,Vector<>,Boolean)` | Executes StoreExperience for the FinancialSACAgent. |
| `Train` |  |
| `UpdateTargetNetworks(Double)` | Executes UpdateTargetNetworks for the FinancialSACAgent. |

