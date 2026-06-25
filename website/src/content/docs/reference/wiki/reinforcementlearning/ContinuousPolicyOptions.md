---
title: "ContinuousPolicyOptions<T>"
description: "Configuration options for continuous action space policies in reinforcement learning."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.ReinforcementLearning.Policies`

Configuration options for continuous action space policies in reinforcement learning.
Continuous policies output actions as real-valued vectors using Gaussian (normal) distributions.

## For Beginners

Continuous policies are for when your actions are numbers on a scale
rather than discrete choices.

Think of the difference:

- Discrete: "Turn left, right, or go straight" (3 choices)
- Continuous: "Turn the wheel 17.3 degrees" (infinite precision)

Real-world examples:

- Robot arm: How much to rotate each joint (0° to 180°)
- Self-driving car: Steering angle (-30° to +30°), acceleration (-5 to +5 m/s²)
- Temperature control: Set thermostat (60°F to 80°F)

The policy learns a "range of good actions" for each situation:

- Mean: The average/best action to take
- Standard deviation: How much to vary around that (exploration)

During training: Sample actions from this range (adds randomness for exploration)
During evaluation: Use the mean action (most confident choice)

This options class lets you configure the network that learns these action ranges.

## How It Works

Continuous policies are essential for reinforcement learning in environments where actions are
real-valued rather than discrete choices. Common applications include robotic control (joint angles,
velocities, torques), autonomous driving (steering angle, acceleration), and financial trading
(position sizes, portfolio weights). The policy network typically outputs both the mean (μ) and
standard deviation (σ) of a Gaussian distribution for each action dimension, enabling the agent
to express uncertainty and explore through stochastic sampling.

This configuration provides defaults optimized for continuous control tasks, based on best practices
from algorithms like SAC (Soft Actor-Critic), PPO (Proximal Policy Optimization), and TD3 (Twin Delayed
DDPG). The larger default network size [256, 256] compared to discrete policies reflects the higher
complexity typically required for smooth continuous control.

## Properties

| Property | Summary |
|:-----|:--------|
| `ActionSize` | Gets or sets the dimensionality of the continuous action space. |
| `StateSize` | Gets or sets the size of the observation/state space. |

