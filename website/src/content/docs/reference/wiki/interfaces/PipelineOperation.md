---
title: "PipelineOperation"
description: "Represents a single operation in the pipeline schedule."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Interfaces`

Represents a single operation in the pipeline schedule.

## For Beginners

This is one instruction in the schedule, like
"do forward pass on micro-batch #3" or "do backward pass on micro-batch #1".

## How It Works

Zero Bubble schedules split the backward pass into two operations:
BackwardInput (compute activation gradients, on the critical path) and
BackwardWeight (compute weight gradients, can fill bubbles). Traditional
schedules use the combined Backward type.

## Properties

| Property | Summary |
|:-----|:--------|
| `IsCooldown` | Gets whether this is a cooldown operation (part of pipeline drain phase). |
| `IsWarmup` | Gets whether this is a warmup operation (part of pipeline fill phase). |
| `MicroBatchIndex` | Gets the micro-batch index this operation works on. |
| `Type` | Gets the type of pipeline operation (Forward, Backward, BackwardInput, or BackwardWeight). |
| `VirtualStageIndex` | Gets the virtual stage index for multi-stage schedules (0-based within this rank). |

