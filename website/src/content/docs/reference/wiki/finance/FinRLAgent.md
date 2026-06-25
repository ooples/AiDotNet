---
title: "FinRLAgent<T>"
description: "Unified FinRL-style agent that can switch between multiple RL algorithms."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Finance.Trading.Agents`

Unified FinRL-style agent that can switch between multiple RL algorithms.

## For Beginners

FinRL is a unified framework for applying reinforcement
learning to stock trading. It can switch between different RL algorithms (DQN, PPO,
A2C, SAC) while providing a consistent trading interface. Think of it as a toolkit
that lets you train an AI trader that learns buy/sell/hold strategies by practicing
on historical market data, similar to how a game AI learns by playing thousands of games.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FinRLAgent(NeuralNetworkArchitecture<>,TradingAgentOptions<>,FinRLAlgorithm,NeuralNetworkArchitecture<>)` | Initializes a new instance of the FinRLAgent class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Algorithm` | Gets the RL algorithm being used. |
| `FeatureCount` |  |
| `ParameterCount` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyGradients(Vector<>,)` |  |
| `Clone` |  |
| `ComputeGradients(Vector<>,Vector<>,ILossFunction<>)` |  |
| `CreateInnerAgent(NeuralNetworkArchitecture<>,NeuralNetworkArchitecture<>,TradingAgentOptions<>,FinRLAlgorithm)` | Creates the concrete agent for the selected algorithm. |
| `Deserialize(Byte[])` |  |
| `GetModelMetadata` |  |
| `GetOptions` |  |
| `GetParameters` |  |
| `LoadModel(String)` | Executes LoadModel for the FinRLAgent. |
| `RequireSecondary(NeuralNetworkArchitecture<>,FinRLAlgorithm)` | Ensures a secondary (critic) architecture is provided when required. |
| `SaveModel(String)` | Executes SaveModel for the FinRLAgent. |
| `SelectAction(Vector<>,Boolean)` |  |
| `Serialize` |  |
| `SetParameters(Vector<>)` |  |
| `StoreExperience(Vector<>,Vector<>,,Vector<>,Boolean)` | Executes StoreExperience for the FinRLAgent. |
| `Train` |  |

