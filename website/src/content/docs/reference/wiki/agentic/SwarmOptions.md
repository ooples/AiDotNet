---
title: "SwarmOptions"
description: "Settings for a `Swarm`: its identity and the overall step budget shared across all members for a single run."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Agentic.Agents`

Settings for a `Swarm`: its identity and the overall step budget shared across all
members for a single run.

## For Beginners

These are the dials for the whole team-of-peers. The most important is
`MaxIterations`, which caps how many total model calls the swarm may make before it must
stop — this is the safety net that prevents two agents from handing a task back and forth forever.

## Properties

| Property | Summary |
|:-----|:--------|
| `Description` | Gets or sets the swarm's description. |
| `MaxIterations` | Gets or sets the maximum total number of model calls across all members in one run. |
| `Name` | Gets or sets the swarm's name. |
| `Temperature` | Gets or sets the sampling temperature forwarded to each member's model call. |

