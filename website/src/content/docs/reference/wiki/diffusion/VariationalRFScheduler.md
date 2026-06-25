---
title: "VariationalRFScheduler<T>"
description: "Variational Rectified Flow scheduler with learned time-dependent noise injection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Schedulers`

Variational Rectified Flow scheduler with learned time-dependent noise injection.

## For Beginners

This scheduler adds a controlled amount of randomness to rectified
flow sampling. Pure deterministic sampling can sometimes produce less diverse results.
This scheduler adds just enough randomness to improve variety without hurting quality.

## How It Works

Extends rectified flow with a variational formulation that allows controlled stochasticity
at each step. The noise injection level is modulated by a learned schedule, improving
diversity while maintaining sample quality.

## Methods

| Method | Summary |
|:-----|:--------|
| `Step(Vector<>,Int32,Vector<>,,Vector<>)` |  |

