---
title: "StochasticInterpolantScheduler<T>"
description: "Stochastic Interpolant scheduler for generalized flow-based sampling with noise injection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Schedulers`

Stochastic Interpolant scheduler for generalized flow-based sampling with noise injection.

## For Beginners

This is a flexible sampler that can behave like either a flow
model (deterministic, straight paths) or a diffusion model (stochastic, with randomness),
or anything in between. You can tune the "stochasticity dial" to find the sweet spot.

## How It Works

Implements the Stochastic Interpolant framework which unifies flow matching and
score-based diffusion. Uses time-dependent interpolation between data and noise with
optional stochasticity controlled by an auxiliary noise schedule.

Reference: Albergo et al., "Stochastic Interpolants: A Unifying Framework for Flows and Diffusions", 2023

## Methods

| Method | Summary |
|:-----|:--------|
| `Step(Vector<>,Int32,Vector<>,,Vector<>)` |  |

