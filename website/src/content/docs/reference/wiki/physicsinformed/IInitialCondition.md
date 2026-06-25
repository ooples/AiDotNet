---
title: "IInitialCondition<T>"
description: "Defines initial conditions for time-dependent PDEs."
section: "API Reference"
---

`Interfaces` · `AiDotNet.PhysicsInformed.Interfaces`

Defines initial conditions for time-dependent PDEs.

## How It Works

For Beginners:
Initial conditions specify the state of the system at the starting time (t=0).
For example, in a heat equation, you might specify the initial temperature distribution.

## Properties

| Property | Summary |
|:-----|:--------|
| `Name` | Gets the name of the initial condition. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeInitialValue(Vector<>)` | Computes the initial condition value at the given spatial location. |
| `IsAtInitialTime(Vector<>)` | Determines if a point is at the initial time. |

