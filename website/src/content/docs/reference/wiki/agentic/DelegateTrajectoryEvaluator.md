---
title: "DelegateTrajectoryEvaluator"
description: "An `ITrajectoryEvaluator` backed by a user-supplied scoring function — the general-purpose hook for custom rewards (exact-match against a labeled answer, regex/JSON validity, cost penalties, or any combination)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Agentic.SelfImproving`

An `ITrajectoryEvaluator` backed by a user-supplied scoring function — the general-purpose
hook for custom rewards (exact-match against a labeled answer, regex/JSON validity, cost penalties, or any
combination).

## For Beginners

The do-it-yourself grader: you provide a small function that looks at a run and
returns a score, and this turns it into an evaluator the rest of the system can use.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DelegateTrajectoryEvaluator(Func<AgentTrajectory,Double>)` | Initializes a new evaluator from a synchronous scoring function. |

## Methods

| Method | Summary |
|:-----|:--------|
| `EvaluateAsync(AgentTrajectory,CancellationToken)` |  |

