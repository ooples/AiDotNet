---
title: "DiscretizedRFScheduler<T>"
description: "Discretized Rectified Flow scheduler with optimized timestep selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Schedulers`

Discretized Rectified Flow scheduler with optimized timestep selection.

## For Beginners

This scheduler picks the best timesteps to use when converting
the continuous flow path into discrete steps. By choosing timesteps more carefully,
it gets better results with fewer steps compared to uniform spacing.

## How It Works

Uses a learned or heuristic discretization of the continuous rectified flow ODE
that minimizes truncation error. Timesteps are selected to minimize the difference
between discrete and continuous ODE solutions.

## Methods

| Method | Summary |
|:-----|:--------|
| `Step(Vector<>,Int32,Vector<>,,Vector<>)` |  |

