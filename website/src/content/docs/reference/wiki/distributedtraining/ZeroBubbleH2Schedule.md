---
title: "ZeroBubbleH2Schedule<T>"
description: "Implements the Zero Bubble H2 (ZB-H2) pipeline schedule."
section: "API Reference"
---

`Models & Types` · `AiDotNet.DistributedTraining`

Implements the Zero Bubble H2 (ZB-H2) pipeline schedule.

## For Beginners

ZB-H2 is the "maximum throughput" variant. It allows more
micro-batches to be in progress simultaneously (using more memory) to completely
eliminate idle time. If you have enough GPU memory, ZB-H2 gives the best possible
pipeline utilization.

The tradeoff:

- ZB-H1: Same memory as 1F1B, ~1/3 bubble
- ZB-H2: More memory than 1F1B, ~0% bubble (zero idle time)

## How It Works

ZB-H2 achieves true zero pipeline bubble by allowing more in-flight micro-batches
than 1F1B, trading peak memory for throughput. Like ZB-H1, it splits backward into
BackwardInput (B) and BackwardWeight (W), but schedules more aggressively.

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
| `EmitCooldown(List<PipelineOperation>,Int32,Int32,Int32)` | Phase 3: Cooldown — drain remaining B and W passes. |
| `EmitExtendedWarmup(List<PipelineOperation>,Int32)` | Phase 1: Extended warmup — more forward passes to fill pipeline completely. |
| `EmitSteadyState(List<PipelineOperation>,Int32,Int32,Int32)` | Phase 2: Steady state — interleave B, F, W to maintain zero bubble. |
| `EstimateBubbleFraction(Int32,Int32)` |  |
| `GetSchedule(Int32,Int32,Int32)` |  |

