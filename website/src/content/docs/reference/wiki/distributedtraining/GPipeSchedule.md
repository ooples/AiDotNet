---
title: "GPipeSchedule<T>"
description: "Implements the GPipe scheduling strategy: all forward passes first, then all backward passes."
section: "API Reference"
---

`Models & Types` · `AiDotNet.DistributedTraining`

Implements the GPipe scheduling strategy: all forward passes first, then all backward passes.

## For Beginners

GPipe is the straightforward approach:

1. Push ALL micro-batches through the forward pass (left to right through stages)
2. Then push ALL micro-batches through the backward pass (right to left)

This creates a "bubble" where stages are idle during pipeline fill and drain.
With P stages and M micro-batches, the bubble fraction is approximately (P-1)/(P-1+M).

For 4 stages and 4 micro-batches:

The underscores represent idle time (bubble).

## How It Works

GPipe is the simplest pipeline schedule. It executes all forward micro-batches sequentially
through the pipeline, storing all activations, then executes all backward micro-batches
in reverse order.

**Reference:** Huang et al., "GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism", 2019.
https://arxiv.org/abs/1811.06965

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

