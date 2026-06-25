---
title: "ITrajectoryStore"
description: "Stores captured `AgentTrajectory` records so the self-improving layer can replay, evaluate, and learn from past agent runs."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Agentic.SelfImproving`

Stores captured `AgentTrajectory` records so the self-improving layer can replay, evaluate,
and learn from past agent runs.

## For Beginners

This is the logbook of everything the agents have done. Other parts of the
system read from it to grade past runs and figure out how to do better next time.

## Methods

| Method | Summary |
|:-----|:--------|
| `AddAsync(AgentTrajectory,CancellationToken)` | Adds a trajectory and returns its id. |
| `ClearAsync(CancellationToken)` | Removes all stored trajectories. |
| `GetAllAsync(CancellationToken)` | Returns all stored trajectories, oldest first. |
| `GetAsync(String,CancellationToken)` | Gets a trajectory by id, or `null` when not found. |
| `QueryAsync(Func<AgentTrajectory,Boolean>,CancellationToken)` | Returns the trajectories matching a predicate, oldest first. |

