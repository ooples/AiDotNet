---
title: "AgentTrajectory"
description: "A captured record of one agent run — the structured \"trajectory\" the self-improving layer learns from."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Agentic.SelfImproving`

A captured record of one agent run — the structured "trajectory" the self-improving layer learns from.
It pairs what the agent did (the full message transcript, the final answer, step count, token usage) with
an optional quality `Reward` assigned later by an evaluator.

## For Beginners

Think of this as the flight recorder for an agent: it saves exactly what was
said and done on a run, plus (once graded) how good the outcome was. Collect many of these and the system
can learn what works and get better over time.

## How It Works

Trajectories are the raw material for every self-improvement mechanism: continuous evaluation scores them,
learned routers/tool policies train on them, prompt optimizers measure prompts against them, and
reward-filtered trajectories become fine-tuning (LoRA) data. Capturing a run is non-invasive — a
`TracingAgent` wraps any agent and records each run without changing its behavior.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AgentTrajectory(String,String,IReadOnlyList<ChatMessage>,String,Int32,ChatUsage,Nullable<Double>,IReadOnlyDictionary<String,String>)` | Initializes a new trajectory. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AgentName` | Gets the name of the agent that produced the run. |
| `FinalText` | Gets the agent's final answer. |
| `Id` | Gets the unique id for this trajectory. |
| `Iterations` | Gets the number of model calls the run took. |
| `Messages` | Gets the full conversation transcript the run produced. |
| `Metadata` | Gets optional key/value metadata, or `null` when none was supplied. |
| `Reward` | Gets or sets the quality score for this trajectory, assigned by an evaluator (higher is better), or `null` while ungraded. |
| `Usage` | Gets aggregate token usage, or `null` when not reported. |

