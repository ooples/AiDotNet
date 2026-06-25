---
title: "SASolverScheduler<T>"
description: "SA-Solver (Stochastic Adams) scheduler using Adams-Bashforth/Moulton methods for SDE sampling."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Schedulers`

SA-Solver (Stochastic Adams) scheduler using Adams-Bashforth/Moulton methods for SDE sampling.

## For Beginners

SA-Solver is a multistep sampler that reuses previous computations
(like DPM-Solver) but for stochastic (random) sampling. This gives you the diversity
benefits of SDE sampling with the efficiency of multistep methods.

## How It Works

SA-Solver applies Adams-Bashforth (predictor) and Adams-Moulton (corrector) multistep
methods to stochastic differential equation sampling. This enables efficient SDE sampling
with higher-order accuracy using cached function evaluations.

Reference: Xue et al., "SA-Solver: Stochastic Adams Solver for Fast Training of Diffusion Models", NeurIPS 2023

## Methods

| Method | Summary |
|:-----|:--------|
| `Step(Vector<>,Int32,Vector<>,,Vector<>)` |  |

