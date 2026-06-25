---
title: "IEnvironment<T>"
description: "Represents a reinforcement learning environment that an agent interacts with."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Represents a reinforcement learning environment that an agent interacts with.

## For Beginners

An environment is the "world" that the RL agent interacts with. Think of it like a video game:

- The agent sees the current state (like where characters are on screen)
- The agent takes actions (like pressing buttons)
- The environment responds with a new state and a reward (like points scored)
- The episode ends when certain conditions are met (like game over)

This interface ensures all environments work consistently with AiDotNet's RL agents.

## How It Works

This interface defines the standard RL environment contract following the OpenAI Gym pattern.
All state observations and actions use AiDotNet's Vector type for consistency with the rest
of the library's type system.

## Properties

| Property | Summary |
|:-----|:--------|
| `ActionSpaceSize` | Gets the size of the action space (number of possible discrete actions or continuous action dimensions). |
| `IsContinuousActionSpace` | Gets whether the action space is continuous (true) or discrete (false). |
| `ObservationSpaceDimension` | Gets the dimension of the observation space. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Close` | Closes the environment and cleans up resources. |
| `Reset` | Resets the environment to an initial state and returns the initial observation. |
| `Seed(Int32)` | Seeds the random number generator for reproducibility. |
| `Step(Vector<>)` | Takes an action in the environment and returns the result. |

