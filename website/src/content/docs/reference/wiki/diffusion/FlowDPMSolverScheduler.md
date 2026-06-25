---
title: "FlowDPMSolverScheduler<T>"
description: "Flow DPM-Solver scheduler applying DPM-Solver acceleration to rectified flow models."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Schedulers`

Flow DPM-Solver scheduler applying DPM-Solver acceleration to rectified flow models.

## For Beginners

This scheduler makes rectified flow models even faster by
reusing previous computation steps. Instead of simple Euler steps, it uses smarter
math (polynomial extrapolation) to take bigger steps while maintaining quality.

## How It Works

Adapts DPM-Solver's multistep prediction to flow-matching ODE trajectories,
using cached previous velocity predictions for higher-order polynomial extrapolation.
Achieves fewer function evaluations than standard Euler rectified flow.

## Methods

| Method | Summary |
|:-----|:--------|
| `Step(Vector<>,Int32,Vector<>,,Vector<>)` |  |

