---
title: "RectifiedFlowScheduler<T>"
description: "Rectified flow scheduler for straight-path ODE sampling with velocity prediction."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Schedulers`

Rectified flow scheduler for straight-path ODE sampling with velocity prediction.

## For Beginners

Rectified flow draws the straightest possible path from noise to
image. This makes each step more efficient — you need fewer steps because you're not
following a curved path. Used by modern models like FLUX and SD3.

## How It Works

Implements rectified flow sampling where the model predicts velocity v = x_1 - x_0 along
straight paths between noise and data. Uses uniform time discretization for optimal
transport between distributions.

Reference: Liu et al., "Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow", ICLR 2023

## Methods

| Method | Summary |
|:-----|:--------|
| `SetTimesteps(Int32)` |  |
| `Step(Vector<>,Int32,Vector<>,,Vector<>)` |  |

