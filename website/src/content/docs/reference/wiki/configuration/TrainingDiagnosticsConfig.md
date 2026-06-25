---
title: "TrainingDiagnosticsConfig"
description: "Process-global control for training-pipeline diagnostic output (gradient norms, optimizer step traces, tape events)."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Configuration`

Process-global control for training-pipeline diagnostic output
(gradient norms, optimizer step traces, tape events). Mirrors the
three-orthogonal-knobs design of `GpuDiagnosticsConfig`:
environment variable, static configuration, and custom sink.

## How It Works

AiDotNet's training pipeline (`NeuralNetworkBase.TrainWithTape`)
can fail in subtle ways — wrong-direction gradient flow, dropped
layer gradients, optimizer skipping parameters, fused-path bailouts.
These bugs are hard to diagnose without per-parameter gradient
visibility (see github.com/ooples/AiDotNet#1328 for an example where
the model trained but only the bias of the final dense head got
useful gradients). This class provides production-grade hooks so
regression suites and end-user debugging can introspect training
without modifying the library.

**Three orthogonal controls** (all work simultaneously):

## Properties

| Property | Summary |
|:-----|:--------|
| `CurrentStepIndex` | Monotonically incrementing step counter, advanced once per call to `TrainWithTape`. |
| `Level` | Current verbosity level. |
| `PerStepEnabled` | Convenience boolean for the most common case ("turn on per-step gradient diagnostics for one test"). |
| `Sink` | Optional sink that receives diagnostic events when set. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AdvanceStep` | Internal — advances and returns the new step index. |
| `Emit(TrainingDiagnosticEvent)` | Emits a structured diagnostic event respecting the current `Level`. |
| `EmitMessage(TrainingDiagnosticLevel,String)` | Convenience helper for emitting a free-form message at a specific level. |

