---
title: "EvaluationReport"
description: "Aggregate statistics from evaluating a set of trajectories: how many were scored, the reward distribution, and the fraction that met a pass threshold."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Agentic.SelfImproving`

Aggregate statistics from evaluating a set of trajectories: how many were scored, the reward
distribution, and the fraction that met a pass threshold. This is the scoreboard a self-improvement loop
watches to know whether it is getting better.

## For Beginners

The report card for a batch of runs — average score, best and worst, and what
percentage "passed". Compare reports before and after a change to see whether the agents improved.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `EvaluationReport(Int32,Double,Double,Double,Double,Double)` | Initializes a new report. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Count` | Gets the number of trajectories scored. |
| `MaxReward` | Gets the maximum reward (0 when none). |
| `MeanReward` | Gets the mean reward across scored trajectories (0 when none). |
| `MinReward` | Gets the minimum reward (0 when none). |
| `PassRate` | Gets the fraction of trajectories meeting the pass threshold (0–1). |
| `PassThreshold` | Gets the threshold used to compute `PassRate`. |

