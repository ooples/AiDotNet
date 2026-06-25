---
title: "IReplayBuffer<T, TState, TAction>"
description: "Interface for experience replay buffers used in reinforcement learning."
section: "API Reference"
---

`Interfaces` · `AiDotNet.ReinforcementLearning.ReplayBuffers`

Interface for experience replay buffers used in reinforcement learning.

## For Beginners

A replay buffer is like a memory bank for the agent. Instead of learning only from the most
recent experience, the agent stores experiences and learns from random samples of past experiences.
This makes learning more stable and efficient.

Think of it like studying for an exam:

- You don't just study the most recent lesson
- You review random material from throughout the course
- This helps you learn connections between different topics
- And prevents forgetting older material

**Common Buffer Types:**

- **Uniform**: All experiences sampled with equal probability
- **Prioritized**: Important experiences (big errors) sampled more often

## How It Works

Experience replay is a technique where the agent stores past experiences and learns from them
multiple times. This breaks temporal correlations and improves sample efficiency. The replay
buffer stores experience tuples (state, action, reward, next_state, done) and provides random
sampling for training.

## Properties

| Property | Summary |
|:-----|:--------|
| `Capacity` | Gets the maximum capacity of the buffer. |
| `Count` | Gets the current number of experiences in the buffer. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Add(Experience<,,>)` | Adds an experience to the buffer. |
| `CanSample(Int32)` | Checks if the buffer has enough experiences to sample a batch. |
| `Clear` | Clears all experiences from the buffer. |
| `Sample(Int32)` | Samples a batch of experiences from the buffer. |

