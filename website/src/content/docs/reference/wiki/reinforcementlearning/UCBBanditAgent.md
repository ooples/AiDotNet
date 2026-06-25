---
title: "UCBBanditAgent<T>"
description: "Upper Confidence Bound (UCB) Multi-Armed Bandit agent."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ReinforcementLearning.Agents.Bandits`

Upper Confidence Bound (UCB) Multi-Armed Bandit agent.

## For Beginners

UCB balances exploration and exploitation mathematically.
For each action, it calculates an optimistic estimate: average reward + confidence bonus.
The confidence bonus is larger for actions tried fewer times, ensuring under-explored
actions get a fair chance. Think of it as "optimism in the face of uncertainty" - if you
are not sure about an action, assume it could be great until proven otherwise. No epsilon
parameter needed; exploration happens automatically.

## Methods

| Method | Summary |
|:-----|:--------|
| `GetOptions` |  |

