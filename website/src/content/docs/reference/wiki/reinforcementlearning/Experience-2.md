---
title: "Experience<T, TState, TAction>"
description: "Represents a single experience tuple (s, a, r, s', done) for reinforcement learning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ReinforcementLearning.ReplayBuffers`

Represents a single experience tuple (s, a, r, s', done) for reinforcement learning.

## For Beginners

An experience is one step of interaction with the environment.
It contains everything the agent needs to learn from that step:

- **State**: What the situation looked like before the agent acted (like a snapshot)
- **Action**: What the agent decided to do
- **Reward**: The feedback received (positive = good, negative = bad, zero = neutral)
- **NextState**: What the situation looks like after the action
- **Done**: Whether this action ended the episode (game over, goal reached, etc.)

For example, in a maze-solving robot:

- State: Robot's current position and sensor readings
- Action: "move forward" or "turn left"
- Reward: +10 for reaching the exit, -1 for hitting a wall, 0 otherwise
- NextState: Robot's new position after the action
- Done: True if robot reached the exit or got stuck

**Common Type Combinations:**

- `Experience<double, Vector<double>, Vector<double>>` - For continuous actions (e.g., robotic control)
- `Experience<double, Vector<double>, int>` - For discrete actions (e.g., game playing)
- `Experience<float, Tensor<float>, int>` - For image-based states (e.g., Atari games)

## How It Works

An Experience is a fundamental data structure in reinforcement learning that captures a single
interaction between an agent and its environment. It consists of five components: the current state,
the action taken, the reward received, the resulting next state, and a flag indicating whether
the episode has ended. This tuple is used to train reinforcement learning agents in algorithms
like Q-learning, Deep Q-Networks (DQN), PPO, and many others.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `Experience(,,,,Boolean)` | Represents a single experience tuple (s, a, r, s', done) for reinforcement learning. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Priority` | Gets or sets the priority for prioritized experience replay. |

