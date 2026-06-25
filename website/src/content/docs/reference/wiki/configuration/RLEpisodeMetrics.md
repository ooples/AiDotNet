---
title: "RLEpisodeMetrics<T>"
description: "Metrics for a completed RL episode."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Configuration`

Metrics for a completed RL episode.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `RLEpisodeMetrics` | Creates a new instance with default values. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AverageLoss` | Average loss during training in this episode. |
| `AverageRewardRecent` | Running average reward over recent episodes (smoothed metric). |
| `ElapsedTime` | Total time elapsed since training started. |
| `Episode` | The episode number (1-indexed). |
| `Steps` | Number of steps taken in this episode. |
| `TerminatedNaturally` | Whether the episode ended naturally (vs hitting max steps). |
| `TotalReward` | Total reward accumulated in this episode. |

