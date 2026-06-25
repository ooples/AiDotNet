---
title: "RewardClippingConfig<T>"
description: "Configuration for reward clipping."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Configuration`

Configuration for reward clipping.

## How It Works

**For Beginners:** Clipping rewards to a fixed range can stabilize training
when reward magnitudes vary widely. The famous Atari DQN paper clipped rewards to [-1, 1].

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `RewardClippingConfig` | Creates a new instance with default values. |

## Properties

| Property | Summary |
|:-----|:--------|
| `MaxReward` | Maximum reward value after clipping. |
| `MinReward` | Minimum reward value after clipping. |
| `UseClipping` | Whether to clip rewards (vs just scaling them). |

