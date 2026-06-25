---
title: "DPMSolverV3Scheduler<T>"
description: "DPM-Solver v3 scheduler with empirical model statistics for improved convergence."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Schedulers`

DPM-Solver v3 scheduler with empirical model statistics for improved convergence.

## For Beginners

DPM-Solver v3 is a smarter version of DPM-Solver that learns from
how the model typically behaves at each step. By understanding the model's patterns,
it can take better shortcuts, reaching high quality in even fewer steps.

## How It Works

DPM-Solver v3 improves upon v2 by incorporating empirical statistics (mean and variance)
of the model output at each timestep. These statistics are used to better estimate the
true ODE solution, reducing both truncation and discretization errors.

Reference: Zheng et al., "DPM-Solver-v3: Improved Diffusion ODE Solver with Empirical Model Statistics", NeurIPS 2023

## Methods

| Method | Summary |
|:-----|:--------|
| `Step(Vector<>,Int32,Vector<>,,Vector<>)` |  |

