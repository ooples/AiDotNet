---
title: "RainbowDQNOptions<T>"
description: "Configuration options for Rainbow DQN agent."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for Rainbow DQN agent.

## How It Works

Rainbow DQN combines six extensions to DQN:

1. Double Q-learning: Reduces overestimation bias
2. Dueling networks: Separates value and advantage streams
3. Prioritized replay: Samples important experiences more frequently
4. Multi-step learning: Uses n-step returns for better credit assignment
5. Distributional RL: Learns full distribution of returns (C51)
6. Noisy networks: Parameter noise for exploration

## Properties

| Property | Summary |
|:-----|:--------|
| `Optimizer` | The optimizer used for updating network parameters. |

