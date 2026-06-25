---
title: "ZeroBubbleH1Schedule<T>"
description: "Implements the Zero Bubble H1 (ZB-H1) pipeline schedule."
section: "API Reference"
---

`Models & Types` · `AiDotNet.DistributedTraining`

Implements the Zero Bubble H1 (ZB-H1) pipeline schedule.

## For Beginners

In standard 1F1B, the backward pass computes both activation and
weight gradients together. ZB-H1 splits this into two steps. The activation gradient (B)
must be done quickly (the previous stage is waiting), but the weight gradient (W) can wait.
By scheduling W during idle time, we reduce wasted time by ~67% compared to 1F1B.

Think of it like a car wash: the "rinse" (B) must happen right after soap, but "waxing" (W)
can be done whenever there's a free slot.

## How It Works

ZB-H1 splits the backward pass into two independent computations:

- **B (BackwardInput)**: Computes activation gradients (dL/dInput) - on the critical path.
- **W (BackwardWeight)**: Computes weight gradients (dL/dWeights) - can be deferred.

By deferring W to fill pipeline bubbles, ZB-H1 reduces the bubble to approximately
one-third of 1F1B's bubble while maintaining the same peak memory footprint.

**Reference:** Qi et al., "Zero Bubble Pipeline Parallelism", ICLR 2024 Spotlight.
https://arxiv.org/abs/2401.10241

## Properties

| Property | Summary |
|:-----|:--------|
| `Name` |  |
| `VirtualStagesPerRank` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `EmitCooldown(List<PipelineOperation>,Int32,Int32,Int32)` | Phase 3: Cooldown — drain remaining BackwardInput and BackwardWeight passes. |
| `EmitSteadyState(List<PipelineOperation>,Int32,Int32,Int32)` | Phase 2: Steady state — 1F-1B-1W pattern. |
| `EmitWarmupForwards(List<PipelineOperation>,Int32)` | Phase 1: Warmup — forward passes only (same as 1F1B). |
| `EstimateBubbleFraction(Int32,Int32)` |  |
| `GetSchedule(Int32,Int32,Int32)` |  |

