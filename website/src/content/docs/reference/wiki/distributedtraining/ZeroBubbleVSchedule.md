---
title: "ZeroBubbleVSchedule<T>"
description: "Implements the Zero Bubble V (ZB-V) pipeline schedule with 2 virtual stages per rank."
section: "API Reference"
---

`Models & Types` · `AiDotNet.DistributedTraining`

Implements the Zero Bubble V (ZB-V) pipeline schedule with 2 virtual stages per rank.

## For Beginners

ZB-V is the best of both worlds:

- Like Interleaved 1F1B: uses 2 model chunks per GPU to reduce bubble
- Like ZB-H1: splits backward into B (activation gradients) and W (weight gradients)
- Unlike ZB-H2: does NOT use extra memory (same as 1F1B)

The result is zero pipeline bubble with no extra memory cost. The tradeoff is slightly
more communication (each microbatch crosses each GPU twice) and implementation complexity.

Example with 4 GPUs (8 total virtual stages):

- GPU 0: virtual stages 0 and 4
- GPU 1: virtual stages 1 and 5
- GPU 2: virtual stages 2 and 6
- GPU 3: virtual stages 3 and 7

Each microbatch flows: 0->1->2->3->4->5->6->7 (visiting each GPU twice).

## How It Works

ZB-V combines the backward decomposition of ZB-H1/H2 with the virtual stage concept of
Interleaved 1F1B, using exactly V=2 virtual stages per rank. Each rank processes two
non-contiguous model chunks, creating a V-shaped execution pattern that achieves zero
pipeline bubble with the same peak memory as standard 1F1B.

The V-shape comes from the execution pattern on each rank:

- First half: Forward passes fill from top to bottom (forward through virtual stage 0)
- Middle: V-shaped transition from forward to backward
- Second half: Backward passes drain from bottom to top (backward through virtual stage 1)

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
| `EmitCooldown(List<PipelineOperation>,Int32,ZeroBubbleVSchedule<>.ScheduleState)` | Phase 3: Drain remaining BackwardWeight operations. |
| `EmitSteadyState(List<PipelineOperation>,Int32,ZeroBubbleVSchedule<>.ScheduleState)` | Phase 2: Steady state — F0, F1, B1, B0, W interleaving until all F/B complete. |
| `EmitWarmup(List<PipelineOperation>,Int32,Int32,ZeroBubbleVSchedule<>.ScheduleState)` | Phase 1: Warmup — interleaved forwards across both virtual stages (depth-first). |
| `EstimateBubbleFraction(Int32,Int32)` |  |
| `GetSchedule(Int32,Int32,Int32)` |  |

