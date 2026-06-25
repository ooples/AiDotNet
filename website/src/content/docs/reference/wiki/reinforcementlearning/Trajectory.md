---
title: "Trajectory<T>"
description: "Represents a trajectory of experience for on-policy RL algorithms (PPO, A2C, etc.)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ReinforcementLearning.Common`

Represents a trajectory of experience for on-policy RL algorithms (PPO, A2C, etc.).

## For Beginners

A trajectory is like recording a game session. It contains:

- Every state you saw
- Every action you took
- Every reward you got
- Additional info (value estimates, action probabilities)

On-policy algorithms like PPO collect these trajectories, learn from them immediately,
then throw them away and collect new ones. This is different from DQN which stores
experiences in a replay buffer and samples from them multiple times.

## How It Works

A trajectory is a sequence of states, actions, and rewards collected by an agent
interacting with an environment. Unlike experience replay (used in DQN), trajectories
are used immediately for training and then discarded in on-policy algorithms.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `Trajectory` | Initializes an empty trajectory. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Actions` | Actions taken during the trajectory. |
| `Advantages` | Computed advantages (used during training). |
| `Dones` | Whether each step was terminal (episode ended). |
| `Length` | Gets the number of steps in the trajectory. |
| `LogProbs` | Log probabilities of actions taken (for policy gradient). |
| `Returns` | Computed returns (discounted sum of rewards). |
| `Rewards` | Rewards received during the trajectory. |
| `States` | States observed during the trajectory. |
| `Values` | Value estimates for each state (from critic). |

## Methods

| Method | Summary |
|:-----|:--------|
| `AddStep(Vector<>,Vector<>,,,,Boolean)` | Adds a step to the trajectory. |
| `Clear` | Clears the trajectory. |

