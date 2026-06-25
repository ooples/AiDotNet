---
title: "FinancialA2CAgent<T>"
description: "Financial Advantage Actor-Critic (A2C) agent for fast trading policy learning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Finance.Trading.Agents`

Financial Advantage Actor-Critic (A2C) agent for fast trading policy learning.

## For Beginners

The A2C (Advantage Actor-Critic) trading agent uses two
neural networks working together: an "actor" that decides what trades to make, and a
"critic" that evaluates how good those decisions are. The advantage of A2C is that it
learns quickly because the critic provides immediate feedback to the actor after each
trade, rather than waiting for the end result. It is well-suited for fast-paced trading
environments where quick adaptation is important.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FinancialA2CAgent(NeuralNetworkArchitecture<>,NeuralNetworkArchitecture<>,TradingAgentOptions<>)` | Initializes a new instance of the FinancialA2CAgent class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `FeatureCount` |  |
| `ParameterCount` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyGradients(Vector<>,)` | Executes ApplyGradients for the FinancialA2CAgent. |
| `Clone` | Executes Clone for the FinancialA2CAgent. |
| `ComputeGradients(Vector<>,Vector<>,ILossFunction<>)` | Executes ComputeGradients for the FinancialA2CAgent. |
| `Deserialize(Byte[])` | Executes Deserialize for the FinancialA2CAgent. |
| `GetModelMetadata` | Executes GetModelMetadata for the FinancialA2CAgent. |
| `GetOptions` |  |
| `GetParameters` | Executes GetParameters for the FinancialA2CAgent. |
| `LoadModel(String)` | Executes LoadModel for the FinancialA2CAgent. |
| `SampleAction(Vector<>)` | Executes SampleAction for the FinancialA2CAgent. |
| `SaveModel(String)` | Executes SaveModel for the FinancialA2CAgent. |
| `SelectAction(Vector<>,Boolean)` |  |
| `Serialize` | Executes Serialize for the FinancialA2CAgent. |
| `SetParameters(Vector<>)` | Executes SetParameters for the FinancialA2CAgent. |
| `StoreExperience(Vector<>,Vector<>,,Vector<>,Boolean)` | Executes StoreExperience for the FinancialA2CAgent. |
| `Train` |  |

