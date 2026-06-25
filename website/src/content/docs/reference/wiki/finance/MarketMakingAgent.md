---
title: "MarketMakingAgent<T>"
description: "Specialized market making agent using reinforcement learning for optimal quoting."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Finance.Trading.Agents`

Specialized market making agent using reinforcement learning for optimal quoting.

## For Beginners

A market making agent learns to provide liquidity by
continuously placing buy and sell orders (quotes) in the market. It earns money from
the spread between its buy and sell prices while managing the risk of holding inventory.
Using reinforcement learning, it learns when to quote aggressively or conservatively
based on market conditions, volatility, and its current position.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MarketMakingAgent` | Initializes a new instance with paper-default options. |
| `MarketMakingAgent(MarketMakingOptions<>)` | Initializes a new instance from `MarketMakingOptions` alone, building the policy-network architecture from `StateSize` and `ActionSize`. |

## Properties

| Property | Summary |
|:-----|:--------|
| `FeatureCount` |  |
| `ParameterCount` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyGradients(Vector<>,)` | Executes ApplyGradients for the MarketMakingAgent. |
| `Clone` | Executes Clone for the MarketMakingAgent. |
| `ComputeGradients(Vector<>,Vector<>,ILossFunction<>)` | Executes ComputeGradients for the MarketMakingAgent. |
| `CreateArchitectureFromOptions(MarketMakingOptions<>)` | Static factory so the base-constructor initializer can null-check `options` before dereferencing `StateSize` / `ActionSize` — a null at the convenience-ctor path would otherwise NullReferenceException inside the initializer rather than thro… |
| `Deserialize(Byte[])` | Executes Deserialize for the MarketMakingAgent. |
| `EnsureMarketMakingLayers(NeuralNetworkArchitecture<>,Int32,Int32)` | Validates the architecture and creates default market-making layers if needed. |
| `GetModelMetadata` | Executes GetModelMetadata for the MarketMakingAgent. |
| `GetOptions` |  |
| `GetParameters` | Executes GetParameters for the MarketMakingAgent. |
| `LoadModel(String)` | Executes LoadModel for the MarketMakingAgent. |
| `SaveModel(String)` | Executes SaveModel for the MarketMakingAgent. |
| `SelectAction(Vector<>,Boolean)` |  |
| `Serialize` | Executes Serialize for the MarketMakingAgent. |
| `SetParameters(Vector<>)` | Executes SetParameters for the MarketMakingAgent. |
| `StoreExperience(Vector<>,Vector<>,,Vector<>,Boolean)` | Executes StoreExperience for the MarketMakingAgent. |
| `Train` |  |

