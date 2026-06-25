---
title: "TrajectoryEvaluationRunner"
description: "Runs an `ITrajectoryEvaluator` over trajectories, annotating each with its `Reward` and producing an aggregate `EvaluationReport`."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Agentic.SelfImproving`

Runs an `ITrajectoryEvaluator` over trajectories, annotating each with its
`Reward` and producing an aggregate `EvaluationReport`. This is
the continuous-evaluation step that turns raw captured runs into a measurable quality signal.

## For Beginners

Hand it a grader and a batch of recorded runs; it grades each one, writes the
score back onto the run, and hands you a summary report. Run it before and after a change to measure
whether the agents are improving.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TrajectoryEvaluationRunner(ITrajectoryEvaluator)` | Initializes a new runner. |

## Methods

| Method | Summary |
|:-----|:--------|
| `EvaluateAsync(IReadOnlyList<AgentTrajectory>,Double,CancellationToken)` | Scores each trajectory (writing `Reward`) and returns an aggregate report. |
| `EvaluateStoreAsync(ITrajectoryStore,Double,CancellationToken)` | Scores every trajectory in a store and returns an aggregate report. |

