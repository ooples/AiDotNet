---
title: "UniformReplayBuffer<T, TState, TAction>"
description: "A replay buffer that samples experiences uniformly at random."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ReinforcementLearning.ReplayBuffers`

A replay buffer that samples experiences uniformly at random.

## For Beginners

This replay buffer treats all experiences equally - it's like having a bag of memories
and pulling out random ones to learn from. When the buffer is full, the oldest memories
get replaced with new ones.

**Key Properties:**

- **Uniform Sampling**: Every experience has an equal chance of being picked
- **Circular Buffer**: Old experiences are automatically removed when capacity is reached
- **No Prioritization**: Unlike prioritized replay, doesn't favor "important" experiences

**When to Use:**

- Good starting point for most RL algorithms
- Works well when all experiences are roughly equally valuable
- Simpler and faster than prioritized variants

## How It Works

This is the standard replay buffer used in algorithms like DQN. Experiences are stored
in a circular buffer and sampled uniformly at random for training. All experiences have
an equal probability of being selected, regardless of their importance or recency.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `UniformReplayBuffer(Int32,Nullable<Int32>)` | Initializes a new instance of the UniformReplayBuffer class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Capacity` |  |
| `Count` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Add(Experience<,,>)` |  |
| `CanSample(Int32)` |  |
| `Clear` |  |
| `Sample(Int32)` |  |
| `SampleWithIndices(Int32)` | Samples a batch of experiences with their buffer indices. |

## Fields

| Field | Summary |
|:-----|:--------|
| `DefaultCapacity` | Initializes a new instance with default settings. |

