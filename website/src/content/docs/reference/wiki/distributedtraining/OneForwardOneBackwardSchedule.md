---
title: "OneForwardOneBackwardSchedule<T>"
description: "Implements the 1F1B (One-Forward-One-Backward) pipeline schedule."
section: "API Reference"
---

`Models & Types` · `AiDotNet.DistributedTraining`

Implements the 1F1B (One-Forward-One-Backward) pipeline schedule.

## For Beginners

Instead of doing ALL forward passes then ALL backward passes (GPipe),
1F1B interleaves them. This is like a factory where each worker handles their current item
and immediately starts the return processing, rather than waiting for all items to pass through.

Benefits:

- Reduces pipeline bubble from ~50% to ~12-15%
- Limits peak memory to (numStages) stored activations instead of (numMicroBatches)
- More efficient for large numbers of micro-batches

Example with 4 stages and 8 micro-batches:

## How It Works

The 1F1B schedule interleaves forward and backward passes to minimize pipeline bubble
and memory usage. It has three phases:

1. **Warmup**: Each stage executes forward passes to fill the pipeline.

Stage i performs (numStages - 1 - i) forward passes before steady state.

2. **Steady State**: Each stage alternates between one forward and one backward pass.

This keeps all stages busy and limits memory usage to at most (numStages) activations.

3. **Cooldown**: Remaining backward passes drain the pipeline.

**Reference:** Narayanan et al., "PipeDream: Generalized Pipeline Parallelism for DNN Training", SOSP 2019.
https://arxiv.org/abs/1806.03377

## Properties

| Property | Summary |
|:-----|:--------|
| `Name` |  |
| `VirtualStagesPerRank` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `EstimateBubbleFraction(Int32,Int32)` |  |
| `GetSchedule(Int32,Int32,Int32)` |  |

