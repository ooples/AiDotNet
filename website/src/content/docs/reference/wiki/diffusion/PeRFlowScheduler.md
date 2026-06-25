---
title: "PeRFlowScheduler<T>"
description: "PeRFlow (Piecewise Rectified Flow) scheduler for accelerated multi-segment flow sampling."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Schedulers`

PeRFlow (Piecewise Rectified Flow) scheduler for accelerated multi-segment flow sampling.

## For Beginners

PeRFlow breaks the generation process into segments, making each
segment's path as straight as possible. This is like straightening a curved road into
connected straight segments — each segment is easy to traverse quickly.

## How It Works

PeRFlow divides the ODE trajectory into K segments, each with independently rectified
straight paths. By ensuring each segment is straight, the overall trajectory requires
fewer total steps while maintaining generation quality.

Reference: Yan et al., "PeRFlow: Piecewise Rectified Flow as Universal Plug-and-Play Accelerator", 2024

## Methods

| Method | Summary |
|:-----|:--------|
| `Step(Vector<>,Int32,Vector<>,,Vector<>)` |  |

