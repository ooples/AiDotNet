---
title: "DiscretePolicyOptions<T>"
description: "Configuration options for discrete action space policies in reinforcement learning."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.ReinforcementLearning.Policies`

Configuration options for discrete action space policies in reinforcement learning.
Discrete policies select from a finite set of actions using categorical (softmax) distributions.

## For Beginners

Discrete policies are for situations where your AI agent must choose
between specific, separate options rather than continuous values.

Think of it like a video game character deciding between actions:

- Move Left
- Move Right
- Jump
- Duck

The policy learns which action is best in each situation by:

1. Looking at the current state (what's on screen)
2. Calculating probabilities for each action (40% jump, 35% left, 20% right, 5% duck)
3. Choosing an action based on these probabilities

During training, it sometimes picks random actions (exploration) to discover new strategies.
During evaluation/playing, it picks the best action it has learned.

This options class lets you configure:

- How many different actions are available (ActionSize)
- How complex the neural network should be (HiddenLayers)
- How much random exploration to use (ExplorationStrategy)

## How It Works

Discrete policies are fundamental to reinforcement learning in environments with finite action spaces,
such as game playing (left/right/jump), robot arm control with discrete positions, or trading decisions
(buy/sell/hold). The policy network outputs logits (unnormalized log probabilities) for each action,
which are then converted to a probability distribution via softmax. Actions are sampled from this
distribution during training to enable exploration, while the most probable action is typically
selected during evaluation.

This configuration class provides sensible defaults aligned with modern deep reinforcement learning
best practices from libraries like Stable Baselines3 and RLlib. The default epsilon-greedy exploration
strategy balances exploration (trying random actions) with exploitation (using learned policy).

## Properties

| Property | Summary |
|:-----|:--------|
| `ActionSize` | Gets or sets the number of discrete actions available to the agent. |
| `ExplorationStrategy` | Gets or sets the exploration strategy for balancing exploration vs exploitation during training. |
| `HiddenLayers` | Gets or sets the architecture of hidden layers in the policy network. |
| `LossFunction` | Gets or sets the loss function used to train the policy network. |
| `Seed` | Gets or sets the random seed for reproducible training runs. |
| `StateSize` | Gets or sets the size of the observation/state space. |

