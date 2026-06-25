---
title: "IPipelineSchedule<T>"
description: "Defines a scheduling strategy for pipeline parallel training."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Defines a scheduling strategy for pipeline parallel training.

## For Beginners

In pipeline parallelism, multiple stages process data like an
assembly line. A "schedule" decides the order of operations to keep all stages as busy
as possible and minimize idle time ("pipeline bubbles").

Think of it like coordinating workers on an assembly line:

- GPipe: Worker 1 finishes ALL items, then Worker 2 starts ALL items (simple but slow)
- 1F1B: Workers alternate between forward and backward steps (more complex but faster)
- Zero Bubble: Workers split backward into two parts, using the flexible part to fill gaps

## How It Works

Pipeline schedules determine the order in which forward and backward passes execute
across micro-batches and stages. Different schedules trade off memory usage, pipeline
bubble overhead, and implementation complexity.

Schedules fall into two categories:

- **Single-stage**: Each rank owns one contiguous model chunk (GPipe, 1F1B, ZB-H1, ZB-H2).
- **Multi-stage**: Each rank owns V non-contiguous chunks ("virtual stages")

(Interleaved 1F1B, Looped BFS, ZB-V).

## Properties

| Property | Summary |
|:-----|:--------|
| `Name` | Gets the name of the scheduling strategy for diagnostics. |
| `VirtualStagesPerRank` | Gets the number of virtual stages (model chunks) each rank holds. |

## Methods

| Method | Summary |
|:-----|:--------|
| `EstimateBubbleFraction(Int32,Int32)` | Estimates the pipeline bubble fraction for this schedule. |
| `GetSchedule(Int32,Int32,Int32)` | Generates the sequence of operations for a given stage in the pipeline. |

