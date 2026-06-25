---
title: "PrioritizedReplayConfig<T>"
description: "Configuration for prioritized experience replay (PER)."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Configuration`

Configuration for prioritized experience replay (PER).

## How It Works

**For Beginners:** PER samples experiences based on their TD-error (surprise).
High-error experiences are sampled more often because they have more to teach.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PrioritizedReplayConfig` | Creates a new instance with default values. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Alpha` | Alpha parameter controlling prioritization strength (0 = uniform, 1 = full prioritization). |
| `BetaAnnealingSteps` | Steps over which to anneal beta from initial to final. |
| `FinalBeta` | Final beta value (should reach 1.0 by end of training). |
| `InitialBeta` | Initial beta for importance sampling correction. |
| `PriorityEpsilon` | Small constant added to priorities to prevent zero probabilities. |

