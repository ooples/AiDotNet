---
title: "CQLOptions<T>"
description: "Configuration options for Conservative Q-Learning (CQL) agent."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for Conservative Q-Learning (CQL) agent.

## For Beginners

CQL is designed for learning from logged data without trying new actions.
This is useful when you have historical data but can't experiment in the real environment
(e.g., medical treatment, autonomous driving).

Key innovation:

- **Conservative Q-Learning**: Penalizes Q-values for unseen actions to prevent overoptimistic estimates
- **Offline Learning**: No environment interaction during training

Think of it like learning to drive from dashcam footage - you can't try new maneuvers,
so you need to be conservative about what you haven't seen.

Based on SAC architecture with conservative regularization.

## How It Works

CQL is an offline RL algorithm that learns from fixed datasets without environment interaction.
It addresses overestimation by adding a conservative penalty to Q-values.

