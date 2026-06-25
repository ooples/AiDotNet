---
title: "LSTDAgent<T>"
description: "LSTD (Least-Squares Temporal Difference) agent using direct solution for value function weights."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ReinforcementLearning.Agents.AdvancedRL`

LSTD (Least-Squares Temporal Difference) agent using direct solution for value function weights.

## For Beginners

LSTD computes the value function weights directly using
matrix operations instead of iterative updates. While standard TD learning takes many
small steps toward the answer, LSTD solves for the answer in one computation (like
solving a system of equations). This is much more sample-efficient but uses more memory
and compute per update. Best for problems with linear function approximation and
moderate state spaces.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LSTDAgent` | Initializes a new instance with default settings. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetOptions` |  |

