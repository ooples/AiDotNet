---
title: "PlanCache"
description: "Disk-backed store for compiled inference plans."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks`

Disk-backed store for compiled inference plans. Persists the traced plan after the
first compilation so subsequent process starts load the pre-compiled plan instead
of re-tracing + re-compiling. Directly wraps Tensors' `CompiledPlanLoader`
and `CancellationToken)`.

## How It Works

PyTorch-parity equivalent: `torch.jit.save(traced_module, path)` +
`torch.jit.load(path)`. The facade integration is opt-in via
`AiModelBuilder.ConfigurePlanCaching(directory)`; once configured, save/load
is transparent to the caller.

Plans are keyed by (modelTypeName, T, structureVersion, inputShapeHash,
hardwareFingerprint). Plans compiled on one host cannot be loaded on a host with
a different `PlanCompatibilityInfo` — Tensors rejects the load and we
fall through to a fresh compile.

## Properties

| Property | Summary |
|:-----|:--------|
| `Current` | The currently-active plan cache, or null if caching is disabled. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetPlanPath(String,Type,Int32,Int32[])` | Computes a stable filename for a plan identified by model type, element type, structure version, and input shape. |
| `SaveInferenceAsync(ICompiledPlan<>,String,Int32,Int32[],CancellationToken)` | Persists a compiled plan to disk via atomic write (tmp-file + rename) so a crash partway through doesn't leave a corrupt cache entry. |
| `TryLoadInferenceAsync(String,Int32,Int32[],IEngine,CancellationToken)` | Attempts to load a pre-compiled inference plan from disk. |

