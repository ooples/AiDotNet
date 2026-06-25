---
title: "DynaQOptions<T>"
description: "Configuration options for Dyna-Q reinforcement learning agents."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for Dyna-Q reinforcement learning agents.

## For Beginners

Dyna-Q is like an agent that not only learns from real experience but also
"imagines" additional experiences using a model of the environment. The planning
steps parameter controls how many imagined experiences the agent uses per real step.

## How It Works

Dyna-Q combines model-free Q-learning with a learned environment model to perform
additional planning steps, accelerating learning from limited real experience.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DynaQOptions` | Initializes a new instance with default values. |
| `DynaQOptions(DynaQOptions<>)` | Initializes a new instance by copying values from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ActionSize` | Number of discrete actions available to the agent. |
| `PlanningSteps` | Number of planning steps per real environment step. |
| `StateSize` | Dimension of the environment state vector. |

