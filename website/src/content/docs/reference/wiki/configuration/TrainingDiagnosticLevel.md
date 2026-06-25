---
title: "TrainingDiagnosticLevel"
description: "Verbosity level for training-pipeline diagnostic output (gradient norms, optimizer step traces, tape replay events, etc.)."
section: "API Reference"
---

`Enums` · `AiDotNet.Configuration`

Verbosity level for training-pipeline diagnostic output (gradient
norms, optimizer step traces, tape replay events, etc.).

## For Beginners

Think of this like log levels in any
logging framework: `Silent` is OFF, `Minimal`
is "just headline events", `Verbose` is "everything
per-batch", `PerStep` is "per-parameter granularity —
expensive but exhaustive".

## How It Works

Companion to `GpuDiagnosticLevel`. AiDotNet's training
pipeline (TrainWithTape, fused-optimizer fast path, gradient-tape
backward) can fail in subtle ways — wrong-direction gradient flow,
dropped layer gradients, optimizer skipping parameters. This enum
gives consumers fine-grained control over training-side diagnostic
output so production code can keep it silent and regression tests
can opt in to detailed traces.

## Fields

| Field | Summary |
|:-----|:--------|
| `Minimal` | Headline events only: loss value per Train call, optimizer step failure, gradient explosion/vanish warnings. |
| `PerStep` | Per-parameter granularity: every parameter tensor's gradient L2 norm after backward, fused-optimizer fast-path hit/miss, scheduler ticks. |
| `Silent` | No training-pipeline diagnostic output. |
| `Verbose` | Per-batch trace: tape forward/backward boundaries, loss, aggregate gradient norm, optimizer step result. |

