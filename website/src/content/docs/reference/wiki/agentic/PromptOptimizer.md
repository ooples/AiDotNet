---
title: "PromptOptimizer<T>"
description: "Selects the best system prompt for an agent by measuring candidate prompts against a labeled eval set — a DSPy-like, evaluation-driven prompt search."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Agentic.SelfImproving`

Selects the best system prompt for an agent by measuring candidate prompts against a labeled eval set —
a DSPy-like, evaluation-driven prompt search. Each candidate builds an agent, runs the eval cases, and is
scored; the highest-scoring prompt wins.

## For Beginners

Instead of guessing which wording works best, you give the optimizer a few
prompt options and a set of practice questions with answers. It tries each prompt on every question, counts
how many it gets right, and hands back the winner.

## How It Works

The optimizer is the *selection* half; the candidate prompts are the *search* half and can
come from anywhere — a hand-written set, an LLM that proposes variations, or AiDotNet's existing
genetic/beam/annealing prompt optimizers feeding their population in. Scoring defaults to a case-insensitive
substring match against the expected answer, overridable with a custom scorer (e.g., reusing an
`ITrajectoryEvaluator`).

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PromptOptimizer(Func<AgentRunResult,PromptEvalCase,Double>)` | Initializes a new prompt optimizer. |

## Methods

| Method | Summary |
|:-----|:--------|
| `OptimizeAsync(IReadOnlyList<String>,Func<String,IAgent<>>,IReadOnlyList<PromptEvalCase>,CancellationToken)` | Evaluates each candidate prompt over the eval set and returns the best. |

