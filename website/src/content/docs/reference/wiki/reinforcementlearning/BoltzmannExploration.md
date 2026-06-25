---
title: "BoltzmannExploration<T>"
description: "Boltzmann (softmax) exploration with temperature-based action selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ReinforcementLearning.Policies.Exploration`

Boltzmann (softmax) exploration with temperature-based action selection.
Uses temperature parameter to control exploration: higher temperature = more random.
Action probability: P(a) = exp(Q(a)/τ) / Σ exp(Q(a')/τ)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `BoltzmannExploration(Double,Double,Double)` | Initializes a new instance of the Boltzmann exploration strategy. |

## Properties

| Property | Summary |
|:-----|:--------|
| `CurrentTemperature` | Gets the current temperature value. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetExplorationAction(Vector<>,Vector<>,Int32,Random)` | Applies Boltzmann (softmax) exploration to select an action. |
| `Reset` | Resets the temperature to its initial value. |
| `Update` | Updates the temperature using exponential decay. |

