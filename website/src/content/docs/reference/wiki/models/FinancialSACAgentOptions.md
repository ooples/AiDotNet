---
title: "FinancialSACAgentOptions<T>"
description: "Configuration options for the Financial SAC (Soft Actor-Critic) trading agent."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for the Financial SAC (Soft Actor-Critic) trading agent.

## For Beginners

SAC is an off-policy actor-critic algorithm that maximizes
both reward and entropy (exploration). It is well-suited for continuous action
spaces like portfolio weight allocation. These options extend the base trading
agent options with SAC-specific parameters.

## Properties

| Property | Summary |
|:-----|:--------|
| `InitialLogAlpha` | Initial log alpha value for automatic temperature tuning. |
| `TargetEntropyRatio` | Target entropy ratio relative to action dimension. |

