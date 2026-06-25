---
title: "ITrajectoryEvaluator"
description: "Scores a captured `AgentTrajectory`, producing the reward signal the self-improving layer optimizes against (higher is better)."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Agentic.SelfImproving`

Scores a captured `AgentTrajectory`, producing the reward signal the self-improving layer
optimizes against (higher is better). This is the seam between "how good was that run?" and every learning
mechanism that consumes the answer.

## For Beginners

A grader. You hand it one recorded run and it returns a score saying how good
the outcome was. Collect scores across many runs and the system can tell which behaviors to reinforce.

## How It Works

Evaluators range from simple (exact-match against a known answer, length/cost penalties) to sophisticated
(an LLM-as-judge, or an adapter over the reasoning reward models). Keeping the contract trajectory-native
means routing/prompt-optimization/fine-tuning all share one definition of quality and can be swapped
without touching the learners.

## Methods

| Method | Summary |
|:-----|:--------|
| `EvaluateAsync(AgentTrajectory,CancellationToken)` | Scores a trajectory. |

