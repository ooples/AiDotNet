---
title: "RLEvaluationConfig"
description: "Configuration for evaluation during training."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Configuration`

Configuration for evaluation during training.

## How It Works

**For Beginners:** Evaluation runs your agent without exploration
to measure true performance. This gives you an unbiased estimate of how well
the agent would perform when deployed.

## Properties

| Property | Summary |
|:-----|:--------|
| `Deterministic` | Whether to use deterministic actions during evaluation. |
| `EvaluateEveryEpisodes` | Evaluate every N episodes. |
| `EvaluationEpisodes` | Number of episodes to run during each evaluation. |

