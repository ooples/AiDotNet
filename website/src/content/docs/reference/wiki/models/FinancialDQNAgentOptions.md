---
title: "FinancialDQNAgentOptions<T>"
description: "Configuration options for the Financial DQN (Deep Q-Network) trading agent."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for the Financial DQN (Deep Q-Network) trading agent.

## For Beginners

DQN is a value-based RL algorithm suited for discrete
action spaces (e.g., buy/hold/sell). These options extend the base trading
agent options with DQN-specific parameters.

## Properties

| Property | Summary |
|:-----|:--------|
| `UseDoubleDQN` | Whether to use double DQN for more stable Q-value estimates. |
| `UseDuelingNetwork` | Whether to use dueling network architecture. |

