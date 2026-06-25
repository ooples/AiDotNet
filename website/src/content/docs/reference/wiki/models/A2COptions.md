---
title: "A2COptions<T>"
description: "Configuration options for Advantage Actor-Critic (A2C) agents."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for Advantage Actor-Critic (A2C) agents.

## For Beginners

A2C learns two things simultaneously:

- **Actor (Policy)**: What action to take in each state
- **Critic (Value Function)**: How good each state is

The critic helps the actor learn faster by providing better feedback than just rewards alone.
Think of the critic as a coach giving targeted advice rather than just "good" or "bad".

A2C is the foundation for many modern RL algorithms including PPO.

## How It Works

A2C is a synchronous version of A3C that is simpler and often more sample-efficient.
It combines policy gradients with value function learning for stable, efficient training.

