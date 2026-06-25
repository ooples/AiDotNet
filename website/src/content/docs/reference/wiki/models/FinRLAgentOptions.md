---
title: "FinRLAgentOptions<T>"
description: "Configuration options for the FinRL unified trading agent."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for the FinRL unified trading agent.

## For Beginners

FinRL is a unified framework that can switch between
multiple RL algorithms (DQN, PPO, A2C, SAC). These options extend the base
trading agent options with algorithm selection and unified configuration.

## Properties

| Property | Summary |
|:-----|:--------|
| `AutoSelectAlgorithm` | Whether to automatically select the best algorithm based on the action space. |
| `UseEnsemble` | Whether to use ensemble of multiple algorithms. |

