---
title: "SupervisorOptions"
description: "Settings for a `SupervisorAgent`: its identity, an optional override of the routing system prompt, and the coordinator's loop/sampling budget."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Agentic.Agents`

Settings for a `SupervisorAgent`: its identity, an optional override of the routing
system prompt, and the coordinator's loop/sampling budget.

## For Beginners

These are the dials for the "team lead". The one you'll most often touch is
`SystemPrompt` if you want to change how the lead decides who does what; otherwise the
defaults give you a working team out of the box.

## How It Works

Every value is nullable and falls back to a sensible default. In particular, leaving
`SystemPrompt``null` lets the supervisor auto-generate a routing prompt that lists
its workers and their specialties — the zero-config path.

## Properties

| Property | Summary |
|:-----|:--------|
| `Description` | Gets or sets the supervisor's description. |
| `MaxIterations` | Gets or sets the maximum number of coordinator model calls in one run (each handoff round is one call). |
| `Name` | Gets or sets the supervisor's name. |
| `SystemPrompt` | Gets or sets an explicit routing system prompt. |
| `Temperature` | Gets or sets the sampling temperature for the coordinator. |

