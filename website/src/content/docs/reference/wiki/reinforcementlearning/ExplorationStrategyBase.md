---
title: "ExplorationStrategyBase<T>"
description: "Abstract base class for exploration strategy implementations."
section: "API Reference"
---

`Base Classes` · `AiDotNet.ReinforcementLearning.Policies.Exploration`

Abstract base class for exploration strategy implementations.
Provides common functionality for noise generation and action clamping.

## Methods

| Method | Summary |
|:-----|:--------|
| `BoxMullerSample(Random)` | Generates a standard normal random sample using the Box-Muller transform. |
| `ClampAction(Vector<>,Double,Double)` | Clamps all elements of an action vector to a specified range. |
| `GetExplorationAction(Vector<>,Vector<>,Int32,Random)` | Modifies or replaces the policy's action for exploration. |
| `Reset` | Resets internal state (e.g., for new episodes or training sessions). |
| `Update` | Updates internal parameters (e.g., epsilon decay, noise reduction). |
| `ValidateActionSize(Int32,Int32,String)` | Validates that an action vector has the expected size. |

## Fields

| Field | Summary |
|:-----|:--------|
| `NumOps` | Numeric operations helper for type-agnostic calculations. |

