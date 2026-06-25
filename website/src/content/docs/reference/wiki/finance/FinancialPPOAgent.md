---
title: "FinancialPPOAgent<T>"
description: "Financial Proximal Policy Optimization (PPO) agent for robust trading."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Finance.Trading.Agents`

Financial Proximal Policy Optimization (PPO) agent for robust trading.

## For Beginners

The PPO (Proximal Policy Optimization) trading agent is one
of the most reliable RL algorithms for trading. It prevents the agent from making too
large a policy change in any single update, which keeps learning stable. Think of it
as a cautious trader who adjusts their strategy gradually rather than making radical
shifts. PPO balances exploration (trying new strategies) with exploitation (sticking
with what works), making it robust for financial applications.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FinancialPPOAgent(NeuralNetworkArchitecture<>,NeuralNetworkArchitecture<>,TradingAgentOptions<>)` | Initializes a new instance of the FinancialPPOAgent class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `FeatureCount` |  |
| `ParameterCount` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyGradients(Vector<>,)` | Executes ApplyGradients for the FinancialPPOAgent. |
| `Clone` | Executes Clone for the FinancialPPOAgent. |
| `ComputeGradients(Vector<>,Vector<>,ILossFunction<>)` | Executes ComputeGradients for the FinancialPPOAgent. |
| `Deserialize(Byte[])` | Executes Deserialize for the FinancialPPOAgent. |
| `GetModelMetadata` | Executes GetModelMetadata for the FinancialPPOAgent. |
| `GetOptions` |  |
| `GetParameters` | Executes GetParameters for the FinancialPPOAgent. |
| `LoadModel(String)` | Executes LoadModel for the FinancialPPOAgent. |
| `SampleCategorical(Vector<>)` | Executes SampleAction for the FinancialPPOAgent. |
| `SaveModel(String)` | Executes SaveModel for the FinancialPPOAgent. |
| `SelectAction(Vector<>,Boolean)` |  |
| `Serialize` | Executes Serialize for the FinancialPPOAgent. |
| `SetParameters(Vector<>)` | Executes SetParameters for the FinancialPPOAgent. |
| `StoreExperience(Vector<>,Vector<>,,Vector<>,Boolean)` | Executes StoreExperience for the FinancialPPOAgent. |
| `Train` |  |

