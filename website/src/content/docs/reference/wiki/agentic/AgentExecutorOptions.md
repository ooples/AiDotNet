---
title: "AgentExecutorOptions"
description: "Settings for an `AgentExecutor`: identity, system prompt, the tool-loop budget, and the sampling knobs forwarded to each model call."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Agentic.Agents`

Settings for an `AgentExecutor`: identity, system prompt, the tool-loop budget, and the
sampling knobs forwarded to each model call.

## For Beginners

These are the dials for one agent. The most useful ones are
`SystemPrompt` (the agent's standing instructions / persona) and `MaxIterations`
(how many think→use-tool→think rounds it may take before it must answer).

## How It Works

Every value is nullable and defaults to a sensible behavior when left `null`, following the
library-wide options pattern (zero-config by default, fully overridable). The executor applies the
documented defaults internally, so the common case is `new AgentExecutor<float>(client)`.

## Properties

| Property | Summary |
|:-----|:--------|
| `Description` | Gets or sets the agent's one-line specialty description. |
| `MaxIterations` | Gets or sets the maximum number of model calls in one run (each tool round costs one call). |
| `MaxOutputTokens` | Gets or sets the per-call output-token cap forwarded to each model call. |
| `Name` | Gets or sets the agent's name (used by coordinators and when the agent is surfaced as a tool). |
| `SystemPrompt` | Gets or sets the system prompt prepended to every run (the agent's standing instructions/persona). |
| `Temperature` | Gets or sets the sampling temperature forwarded to each model call. |
| `ToolChoice` | Gets or sets how eagerly the model may use the agent's tools when tools are present. |

## Fields

| Field | Summary |
|:-----|:--------|
| `DefaultMaxIterations` | The default maximum number of model calls in a single run when `MaxIterations` is unset. |

