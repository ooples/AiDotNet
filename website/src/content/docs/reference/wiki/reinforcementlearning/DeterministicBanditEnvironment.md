---
title: "DeterministicBanditEnvironment<T>"
description: "A deterministic multi-armed bandit environment for testing purposes."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ReinforcementLearning.Environments`

A deterministic multi-armed bandit environment for testing purposes.

## For Beginners

A bandit is like a slot machine. Each "arm" (action) has a fixed reward.
This deterministic version always gives the same reward for the same action,
making it perfect for testing - you always know what reward to expect.

## How It Works

This environment provides a simple bandit setting where each action has a fixed deterministic reward.
It's useful for testing RL data loaders and agents with predictable outcomes.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DeterministicBanditEnvironment(Int32,Int32,Int32,Int32)` | Initializes a new instance of the `DeterministicBanditEnvironment` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ActionSpaceSize` |  |
| `IsContinuousActionSpace` |  |
| `ObservationSpaceDimension` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Close` |  |
| `Reset` |  |
| `Seed(Int32)` |  |
| `Step(Vector<>)` |  |

