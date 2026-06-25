---
title: "LearnedAgentRouter<T>"
description: "A router that learns, from graded `AgentTrajectory` history, which candidate agent performs best and routes new requests accordingly — a contextual reward-weighted bandit."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Agentic.SelfImproving`

A router that learns, from graded `AgentTrajectory` history, which candidate agent performs
best and routes new requests accordingly — a contextual reward-weighted bandit. As more rewarded runs
accumulate, the routing improves, so the orchestration gets better at its task over time without a
hand-tuned router.

## For Beginners

Imagine a dispatcher who remembers which worker did best on each kind of job.
Given a new job it sends it to the worker with the best track record (occasionally trying others to keep
learning). Feed it the graded history and it gets smarter about who to pick.

## How It Works

Because it is an `IAgent`, the router drops in anywhere an agent is expected and composes
with tracing (so its own routed runs feed back into the trajectory store, closing the self-improvement
loop). It exploits the highest mean-reward agent for the request's context, explores unseen agents first
(optimistic), and takes a random agent with probability `explorationRate` to keep discovering. An
optional context key lets it learn different best-agents for different kinds of task.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LearnedAgentRouter(IReadOnlyList<IAgent<>>,Double,Func<IReadOnlyList<ChatMessage>,String>,Nullable<Int32>)` | Initializes a new learned router. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Description` |  |
| `Name` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `LearnFrom(IEnumerable<AgentTrajectory>)` | Updates the router's policy from graded trajectories. |
| `RunAsync(IReadOnlyList<ChatMessage>,CancellationToken)` |  |
| `SelectAgentName(IReadOnlyList<ChatMessage>)` | Returns the name of the agent the router would choose for the given request, without running it (deterministic when `explorationRate` is 0). |

