---
title: "SoftmaxPolicyRouter<T>"
description: "A routing policy over candidate agents, trained by REINFORCE: it keeps a learnable preference weight per agent (optionally per context), selects via a softmax over those weights, and updates them with a policy-gradient step from observed re…"
section: "API Reference"
---

`Models & Types` · `AiDotNet.Agentic.SelfImproving`

A routing policy over candidate agents, trained by REINFORCE: it keeps a learnable preference weight per
agent (optionally per context), selects via a softmax over those weights, and updates them with a
policy-gradient step from observed rewards. Compared with the simpler mean-reward bandit
(`LearnedAgentRouter`), this learns a stochastic policy that handles non-stationary rewards
and keeps exploring in proportion to its current confidence.

## For Beginners

A dispatcher that keeps a "preference score" for each worker and nudges those
scores up or down based on how well the chosen worker did versus the recent average. Over time it sends
more work to the workers that do better, while still occasionally trying others.

## How It Works

The update for an observed (agent, reward) is the REINFORCE rule with a running-mean baseline:
`w[a'] += lr · (reward − baseline) · (1[a'=chosen] − π(a'))` for every candidate `a'`. As a
result, agents that earn above-baseline reward gain probability mass. As an `IAgent` it
composes with tracing, closing the loop.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SoftmaxPolicyRouter(IReadOnlyList<IAgent<>>,Double,Func<IReadOnlyList<ChatMessage>,String>,Nullable<Int32>)` | Initializes a new policy router. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Description` |  |
| `Name` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `LearnFrom(IEnumerable<AgentTrajectory>)` | Trains the policy from graded trajectories (each whose agent is a candidate and whose reward is set). |
| `ProbabilityOf(String,IReadOnlyList<ChatMessage>)` | Gets the current policy probability of choosing an agent for a context. |
| `RunAsync(IReadOnlyList<ChatMessage>,CancellationToken)` |  |
| `SelectBestAgentName(IReadOnlyList<ChatMessage>)` | Returns the highest-probability agent for a context (deterministic, the exploit choice). |
| `Update(IReadOnlyList<ChatMessage>,String,Double)` | Applies a single REINFORCE update for an observed (agent, reward) in the given context. |

