---
title: "LoopedBFSSchedule<T>"
description: "Implements the Looped BFS (Breadth-First Schedule) pipeline schedule with multiple virtual stages per rank."
section: "API Reference"
---

`Models & Types` · `AiDotNet.DistributedTraining`

Implements the Looped BFS (Breadth-First Schedule) pipeline schedule with multiple virtual stages per rank.

## For Beginners

Imagine a factory with two assembly stations per worker (V=2).
Depth-first (Interleaved 1F1B) means: finish one product at both stations before starting the next.
Breadth-first (Looped BFS) means: run all products through station 1, then all through station 2.

Looped BFS tends to have slightly higher pipeline utilization in some configurations because
it minimizes the number of times data needs to cross between physical ranks. However, it
may have higher peak memory usage since more microbatches are in flight at each virtual stage.

Example with 4 GPUs, V=2 (8 total chunks):

- GPU 0: chunks 0 and 4
- GPU 1: chunks 1 and 5
- GPU 2: chunks 2 and 6
- GPU 3: chunks 3 and 7

Looped BFS processes ALL microbatches through chunks 0-3 first (loop 1),
then ALL microbatches through chunks 4-7 (loop 2).

## How It Works

Looped BFS, like Interleaved 1F1B, assigns V non-contiguous model chunks ("virtual stages")
to each rank. Rank i holds chunks {i, i+P, i+2P, ...} where P is the number of physical ranks.

The key difference from Interleaved 1F1B is the scheduling priority:

- **Interleaved 1F1B (Depth-First)**: Prioritizes the **earlier microbatch**. If microbatch 0

is ready for virtual stages 0 and 1, it runs stage 0 for microbatch 0 first.

- **Looped BFS (Breadth-First)**: Prioritizes the **earlier virtual stage**. If microbatches 0

and 1 are ready for virtual stage 0, it processes them both before moving to stage 1.

**Reference:** Lamy-Poirier, "Breadth-First Pipeline Parallelism", 2022.
https://arxiv.org/abs/2211.05953

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LoopedBFSSchedule(Int32)` | Creates a new Looped BFS schedule. |

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

