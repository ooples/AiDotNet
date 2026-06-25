---
title: "EveryVisitMonteCarloAgent<T>"
description: "Every-Visit Monte Carlo agent that updates all visits to states in an episode."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ReinforcementLearning.Agents.MonteCarlo`

Every-Visit Monte Carlo agent that updates all visits to states in an episode.

## For Beginners

Every-Visit Monte Carlo learns by playing complete episodes
(from start to finish) and then averaging the total reward received. Unlike First-Visit MC
which only counts the first time a state is seen, this counts every visit. Think of it
like a student who reviews every practice problem, not just the first attempt. This gives
more data points per episode but with potentially correlated samples. Good for episodic
tasks like board games where you learn from complete games.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `EveryVisitMonteCarloAgent` | Initializes a new instance with default settings. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clone` | Creates a deep copy of the agent, including all Q-table entries. |
| `ComputeAverage(List<>)` | Computes the average of a list of returns. |
| `GetActionIndex(Vector<>)` | Gets the index of the selected action from a one-hot encoded action vector. |
| `GetOptions` |  |
| `SetParameters(Vector<>)` | Sets parameters. |
| `VectorToStateKey(Vector<>)` | Converts a state vector to a string key for the Q-table. |

