---
title: "TargetNetworkConfig<T>"
description: "Configuration for target network updates in DQN-family algorithms."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Configuration`

Configuration for target network updates in DQN-family algorithms.

## How It Works

**For Beginners:** DQN uses two networks - a main network for selecting actions
and a target network for computing stable Q-value targets. This configuration
controls how often and how the target network gets updated.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TargetNetworkConfig` | Creates a new instance with default values. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Tau` | Tau parameter for soft updates (0 to 1). |
| `UpdateFrequency` | Update target network every N steps. |
| `UseSoftUpdate` | Whether to use soft updates (Polyak averaging) instead of hard updates. |

