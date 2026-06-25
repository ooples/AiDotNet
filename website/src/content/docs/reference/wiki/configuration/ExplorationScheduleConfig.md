---
title: "ExplorationScheduleConfig<T>"
description: "Configuration for exploration schedule (epsilon decay for epsilon-greedy)."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Configuration`

Configuration for exploration schedule (epsilon decay for epsilon-greedy).

## How It Works

**For Beginners:** The agent needs to explore early in training (try random actions)
but exploit more later (use learned policy). This schedule controls that transition.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ExplorationScheduleConfig` | Creates a new instance with default values. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DecaySteps` | Number of steps over which to decay from initial to final epsilon. |
| `DecayType` | Type of decay schedule. |
| `FinalEpsilon` | Final exploration rate (0.01 = mostly learned policy). |
| `InitialEpsilon` | Initial exploration rate (1.0 = fully random). |

