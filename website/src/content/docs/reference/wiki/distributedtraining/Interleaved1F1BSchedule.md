---
title: "Interleaved1F1BSchedule<T>"
description: "Implements the Interleaved 1F1B pipeline schedule with multiple virtual stages per rank."
section: "API Reference"
---

`Models & Types` · `AiDotNet.DistributedTraining`

Implements the Interleaved 1F1B pipeline schedule with multiple virtual stages per rank.

## For Beginners

Standard 1F1B gives each GPU one big chunk of the model.
Interleaved 1F1B gives each GPU V smaller, evenly-spaced chunks instead.

Example with 4 GPUs, V=2 (8 total chunks):

- GPU 0: chunks 0 and 4
- GPU 1: chunks 1 and 5
- GPU 2: chunks 2 and 6
- GPU 3: chunks 3 and 7

This means each microbatch visits each GPU twice (once for each chunk), creating more
opportunities to interleave work and reduce idle time. The bubble shrinks from
~(P-1)/(2M+P-1) to ~(P-1)/(2MV+P-1).

Used in production by Megatron-LM v2 and NVIDIA NeMo.

## How It Works

Interleaved 1F1B assigns V non-contiguous model chunks ("virtual stages") to each rank.
Rank i holds chunks {i, i+P, i+2P, ...} where P is the number of physical ranks.
This reduces the pipeline bubble by a factor of V compared to standard 1F1B.

When a microbatch is ready for multiple local virtual stages, Interleaved 1F1B
prioritizes the **earlier microbatch** (depth-first ordering). This is in contrast
to Looped BFS which prioritizes the earlier stage.

**Reference:** Narayanan et al., "Efficient Large-Scale Language Model Training
on GPU Clusters Using Megatron-LM", SC 2021. https://arxiv.org/abs/2104.04473

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `Interleaved1F1BSchedule(Int32)` | Creates a new Interleaved 1F1B schedule. |

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

