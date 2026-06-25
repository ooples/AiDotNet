---
title: "REINFORCEOptions<T>"
description: "Configuration options for REINFORCE agents."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for REINFORCE agents.

## For Beginners

REINFORCE is the "hello world" of policy gradient methods. It's simple but powerful:

- Play an entire episode
- See which actions led to good rewards
- Make those actions more likely in the future

Think of it like learning to play a game: you play a round, see your score,
then adjust your strategy to do better next time.

Simple, but can be slow to learn and high variance.
Modern algorithms like PPO improve on REINFORCE's ideas.

## How It Works

REINFORCE is the simplest policy gradient algorithm. It directly optimizes
the policy by following the gradient of expected returns.

## Properties

| Property | Summary |
|:-----|:--------|
| `ActionSize` | Number of discrete actions available to the agent. |
| `DiscountFactor` | Discount factor (gamma) for future rewards. |
| `HiddenLayers` | Hidden layer sizes for the policy network. |
| `IsContinuous` | Whether the action space is continuous (true) or discrete (false). |
| `LearningRate` | Learning rate for policy gradient updates. |
| `StateSize` | Dimension of the environment state vector. |

