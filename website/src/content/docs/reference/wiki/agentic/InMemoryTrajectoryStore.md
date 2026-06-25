---
title: "InMemoryTrajectoryStore"
description: "A process-local `ITrajectoryStore` that keeps captured runs in memory."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Agentic.SelfImproving`

A process-local `ITrajectoryStore` that keeps captured runs in memory. Ideal for tests and
single-process self-improvement loops; contents are lost when the process exits.

## For Beginners

The simplest logbook, kept in RAM — fast and zero-config, but not saved to
disk. Swap in a durable store when you need trajectories to persist across restarts.

## Methods

| Method | Summary |
|:-----|:--------|
| `AddAsync(AgentTrajectory,CancellationToken)` |  |
| `ClearAsync(CancellationToken)` |  |
| `GetAllAsync(CancellationToken)` |  |
| `GetAsync(String,CancellationToken)` |  |
| `QueryAsync(Func<AgentTrajectory,Boolean>,CancellationToken)` |  |

