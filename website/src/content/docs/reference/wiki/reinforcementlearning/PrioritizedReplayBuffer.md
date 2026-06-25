---
title: "PrioritizedReplayBuffer<T>"
description: "Prioritized experience replay buffer for reinforcement learning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ReinforcementLearning.ReplayBuffers`

Prioritized experience replay buffer for reinforcement learning.

## How It Works

Prioritized replay samples important experiences more frequently based on TD error.
Uses sum tree data structure for efficient sampling.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PrioritizedReplayBuffer` | Initializes a new instance with default settings. |

