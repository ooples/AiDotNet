---
title: "IRLDataLoader<T>"
description: "Interface for data loaders that provide experience data for reinforcement learning."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Interface for data loaders that provide experience data for reinforcement learning.

## For Beginners

Reinforcement learning is learning through trial and error.

**How it works:**

- An agent takes actions in an environment
- The environment returns rewards and new states
- The agent learns to maximize total rewards

**Example: Game Playing**

- Environment: The game (e.g., CartPole, Atari, Chess)
- State: What the agent sees (game screen, piece positions)
- Action: What the agent does (move left, jump, place piece)
- Reward: Score or outcome (+1 for winning, -1 for losing)

**This data loader:**

- Runs episodes in the environment
- Collects experience tuples (state, action, reward, next_state, done)
- Stores them in a replay buffer for training
- Provides batches of experiences for learning

## How It Works

This interface is for reinforcement learning scenarios where an agent interacts with an
environment to collect experience data. The loader manages:

- Environment interactions (stepping through episodes)
- Experience collection and storage
- Replay buffer management for batch sampling

## Properties

| Property | Summary |
|:-----|:--------|
| `CurrentEpisode` | Gets the current episode number (0-indexed). |
| `Environment` | Gets the environment that the agent interacts with. |
| `Episodes` | Gets the total number of episodes to run during training. |
| `MaxStepsPerEpisode` | Gets the maximum number of steps per episode (prevents infinite episodes). |
| `MinExperiencesBeforeTraining` | Gets the minimum number of experiences required before training can begin. |
| `ReplayBuffer` | Gets the replay buffer used for storing and sampling experiences. |
| `TotalSteps` | Gets the total number of steps taken across all episodes. |
| `Verbose` | Gets whether to print training progress to console. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AddExperience(Experience<,Vector<>,Vector<>>)` | Adds an experience to the replay buffer. |
| `CanTrain(Int32)` | Checks if there are enough experiences to begin training. |
| `ResetTraining` | Resets the data loader state (clears buffer, resets counters). |
| `RunEpisode(IRLAgent<>)` | Runs a single episode and collects experiences. |
| `RunEpisodes(Int32,IRLAgent<>)` | Runs multiple episodes and collects experiences. |
| `SampleBatch(Int32)` | Samples a batch of experiences from the replay buffer. |
| `SetSeed(Int32)` | Sets the random seed for reproducible training. |

