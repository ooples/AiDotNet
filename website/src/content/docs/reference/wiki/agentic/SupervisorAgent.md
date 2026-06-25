---
title: "SupervisorAgent<T>"
description: "A coordinator agent that supervises a team of specialized worker `IAgent` instances and routes work to them via native tool-calling."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Agentic.Agents`

A coordinator agent that supervises a team of specialized worker `IAgent` instances and
routes work to them via native tool-calling. Each worker is surfaced as a `transfer_to_<worker>`
handoff tool, so the supervisor decides — turn by turn — which teammate to delegate to, reads their
result, and continues until it produces a final answer.

## For Beginners

Think of a project lead with a team of specialists. You give the lead a
goal; the lead picks the right specialist for each step ("you handle the math", "you write the summary"),
collects their work, and reports back the final result. You only talk to the lead.

## How It Works

Because a `SupervisorAgent` is itself an `IAgent`, supervisors compose:
a top-level supervisor can have other supervisors as workers, forming hierarchical teams. The routing
itself reuses `AgentExecutor`'s tool-calling loop, so there is no bespoke control flow —
delegation is just the coordinator calling tools that happen to be other agents.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SupervisorAgent(IChatClient<>,IReadOnlyList<IAgent<>>,SupervisorOptions)` | Initializes a new supervisor over the supplied worker agents. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Description` |  |
| `Name` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `RunAsync(IReadOnlyList<ChatMessage>,CancellationToken)` |  |
| `RunAsync(String,CancellationToken)` | Runs the supervisor against a single user request. |

