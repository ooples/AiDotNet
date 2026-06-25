---
title: "ChatClientTrajectoryEvaluator<T>"
description: "An `ITrajectoryEvaluator` that scores a run with a model acting as judge (\"LLM-as-judge\"): it shows a judging model the task and the agent's answer (and an optional rubric) and parses a 0–1 score from the reply."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Agentic.SelfImproving`

An `ITrajectoryEvaluator` that scores a run with a model acting as judge ("LLM-as-judge"):
it shows a judging model the task and the agent's answer (and an optional rubric) and parses a 0–1 score
from the reply. This is the general, model-based reward signal the self-improving layer can optimize
against without a hand-written metric.

## For Beginners

Instead of writing rules to grade an answer, you ask another model "how good
is this, from 0 to 1?". This wraps that into a grader the rest of the system uses like any other.

## How It Works

The judge can be any connector — including the in-process `LocalEngineChatClient`, so evaluation runs
fully offline. (The reasoning reward models in `Reasoning/Verification` can likewise be adapted to
this interface; LLM-as-judge is the connector-native default.) Scores are clamped to [0, 1].

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ChatClientTrajectoryEvaluator(IChatClient<>,String)` | Initializes a new LLM-as-judge evaluator. |

## Methods

| Method | Summary |
|:-----|:--------|
| `EvaluateAsync(AgentTrajectory,CancellationToken)` |  |

