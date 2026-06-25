---
title: "EnvironmentDataLoader<T>"
description: "Data loader for reinforcement learning that wraps an environment for experience collection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Loaders.RL`

Data loader for reinforcement learning that wraps an environment for experience collection.

## For Beginners

This is the main way to set up RL training:

**What it does:**

- Wraps your RL environment
- Creates a replay buffer for storing experiences
- Manages episode running during training
- Provides experience batches for learning

## How It Works

EnvironmentDataLoader provides a clean facade for RL training by wrapping an environment
and managing experience collection. Use this with AiModelBuilder for unified training.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `EnvironmentDataLoader(IEnvironment<>,IReplayBuffer<,Vector<>,Vector<>>,Int32,Int32,Int32,Boolean,Nullable<Int32>)` | Initializes a new EnvironmentDataLoader with a custom replay buffer. |
| `EnvironmentDataLoader(IEnvironment<>,Int32,Int32,Int32,Int32,Boolean,Nullable<Int32>)` | Initializes a new EnvironmentDataLoader with the specified environment. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Name` |  |

