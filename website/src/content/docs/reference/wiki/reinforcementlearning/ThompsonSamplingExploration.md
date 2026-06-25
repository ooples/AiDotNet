---
title: "ThompsonSamplingExploration<T>"
description: "Thompson Sampling (Bayesian) exploration for discrete action spaces."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ReinforcementLearning.Policies.Exploration`

Thompson Sampling (Bayesian) exploration for discrete action spaces.
Maintains Beta distributions for each action and samples from posteriors.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ThompsonSamplingExploration(Double,Double)` | Initializes a new instance of the Thompson Sampling exploration strategy. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetExplorationAction(Vector<>,Vector<>,Int32,Random)` | Selects action by sampling from Beta posteriors for each action. |
| `Reset` | Resets all action distributions to prior. |
| `Update` | Updates internal parameters (call UpdateDistribution separately for each action). |
| `UpdateDistribution(Int32,Double)` | Updates the Beta distribution for a specific action based on reward. |

