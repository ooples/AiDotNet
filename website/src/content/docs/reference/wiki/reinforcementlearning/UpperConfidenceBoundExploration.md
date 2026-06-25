---
title: "UpperConfidenceBoundExploration<T>"
description: "Upper Confidence Bound (UCB) exploration for discrete action spaces."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ReinforcementLearning.Policies.Exploration`

Upper Confidence Bound (UCB) exploration for discrete action spaces.
Balances exploration and exploitation using confidence intervals: UCB(a) = Q(a) + c * √(ln(t) / N(a))

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `UpperConfidenceBoundExploration(Double)` | Initializes a new instance of the Upper Confidence Bound exploration strategy. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ExplorationConstant` | Gets the current exploration constant. |
| `TotalSteps` | Gets the total number of steps taken. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetExplorationAction(Vector<>,Vector<>,Int32,Random)` | Selects action using UCB: action with highest Q(a) + c * √(ln(t) / N(a)) |
| `Reset` | Resets action counts and total steps. |
| `Update` | Updates internal parameters (UCB is count-based, no explicit decay). |

