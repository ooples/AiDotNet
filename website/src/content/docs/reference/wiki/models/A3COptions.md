---
title: "A3COptions<T>"
description: "Configuration options for Asynchronous Advantage Actor-Critic (A3C) agents."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for Asynchronous Advantage Actor-Critic (A3C) agents.

## For Beginners

A3C is like having multiple students learn the same subject simultaneously,
each with different experiences. They periodically share what they learned
with a central "teacher" (global network), and everyone benefits from the
combined knowledge.

Key features:

- **Asynchronous**: Multiple agents run in parallel
- **Actor-Critic**: Learns both policy and value function
- **No Replay Buffer**: Uses on-policy learning
- **Diverse Exploration**: Different agents explore different strategies

Famous for: DeepMind's breakthrough paper (2016), enables CPU-only training

## How It Works

A3C runs multiple agents in parallel, each learning from different experiences.
The parallel exploration provides diverse training data and stabilizes learning.

## Properties

| Property | Summary |
|:-----|:--------|
| `Optimizer` | The optimizer used for updating network parameters. |

