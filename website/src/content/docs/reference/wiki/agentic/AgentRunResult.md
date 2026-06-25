---
title: "AgentRunResult"
description: "The outcome of an agent run: the final text answer, the full conversation transcript it produced, how many model calls it took, whether it finished cleanly, and the aggregate token usage."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Agentic.Agents`

The outcome of an agent run: the final text answer, the full conversation transcript it produced,
how many model calls it took, whether it finished cleanly, and the aggregate token usage.

## For Beginners

After an agent runs, this is what you get back. `FinalText` is
the answer most callers want; `Messages` is the full play-by-play if you want to see how it
got there; `Completed` tells you whether it actually finished or ran out of allowed steps.

## How It Works

`Messages` is the complete transcript the agent worked with (system prompt, the inbound
conversation, every assistant turn, and every tool-result message), so a caller can inspect the
agent's reasoning, persist the thread, or feed it into another agent. `Completed` is
`false` only when the agent hit its iteration cap before producing a final (non-tool) answer.

## Properties

| Property | Summary |
|:-----|:--------|
| `AgentName` | Gets the name of the agent that produced the final answer, or `null` when not tracked. |
| `Completed` | Gets a value indicating whether the agent produced a final answer (`true`) or stopped because it reached its iteration cap (`false`). |
| `FinalText` | Gets the agent's final text answer. |
| `Iterations` | Gets the number of model calls the run made (each tool round is one call). |
| `Messages` | Gets the complete conversation transcript the agent produced, including the system prompt (if any), the inbound messages, every assistant turn, and every tool-result message. |
| `Usage` | Gets the aggregate token usage across every model call in the run, or `null` when no call reported usage. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Finished(String,IReadOnlyList<ChatMessage>,Int32,ChatUsage,String)` | Creates a result for a run that produced a final answer. |
| `Stopped(String,IReadOnlyList<ChatMessage>,Int32,ChatUsage,String)` | Creates a result for a run that hit its iteration cap before producing a final answer. |

