---
title: "MixedPrecisionContext"
description: "Manages master weights (FP32) and working weights (FP16) for mixed-precision training."
section: "API Reference"
---

`Models & Types` · `AiDotNet.MixedPrecision`

Manages master weights (FP32) and working weights (FP16) for mixed-precision training.

## For Beginners

Mixed-precision training uses two copies of model parameters:

1. **Master Weights** (FP32):
- High-precision copy of all parameters
- Used for parameter updates to maintain accuracy
- Stored in memory but not used for forward/backward passes

2. **Working Weights** (FP16):
- Low-precision copy used for computation
- Used in forward and backward passes
- Faster and uses less memory
- Synced from master weights before each forward pass

The workflow:

1. Cast master weights (FP32) to working weights (FP16)
2. Forward pass using FP16 weights → faster, less memory
3. Backward pass in FP16 → computes FP16 gradients
4. Cast gradients to FP32 and unscale
5. Update master weights in FP32 → maintains precision
6. Repeat from step 1

This approach combines the speed of FP16 with the numerical stability of FP32.

## How It Works

**Technical Details:** The context maintains:

- Dictionary mapping parameter names to FP32 master copies
- Dictionary mapping parameter names to FP16 working copies
- Synchronization methods to cast between precisions
- Integration with LossScaler for gradient management

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MixedPrecisionContext(MixedPrecisionConfig)` | Initializes a new mixed-precision training context. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Config` | Configuration for mixed-precision training. |
| `IsInitialized` | Whether the context has been initialized with parameters. |
| `LossScaler` | Loss scaler for gradient scaling and overflow detection. |
| `ParameterCount` | Number of parameters managed by this context. |
| `ParameterNames` | Gets the names of all parameters being managed. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CastWeightsToBF16(String)` | Round-trips the master weights through BF16 (truncate-low-16-bits-of-FP32) to emulate a forward pass that would have used BF16 working weights. |
| `CastWeightsToFP16` | Converts master weights (FP32) to working weights (FP16) for forward pass. |
| `ClearFp32SnapshotPool` | Drop the FP32 snapshot pool. |
| `Dispose` | Disposes of the context and releases resources. |
| `EvictFp32Snapshot(Tensor<Single>)` | Evict the cached FP32 master snapshot for `param`. |
| `GetMasterWeights(String)` | Gets the master weights (FP32) for a parameter group. |
| `GetOrCreateFp32Snapshot(Tensor<Single>)` | Get (or grow) the cached FP32 master snapshot for `param`. |
| `GetWorkingWeights(String)` | Gets the working weights (FP16) for a parameter group. |
| `HasFullFP32Precision(Single)` | Returns true if `masterWeight` carries low-mantissa bits that an FP16 / BF16 round-trip would have zeroed out — i.e. |
| `Initialize(Dictionary<String,Vector<Single>>)` | Initializes the context with multiple named parameter groups. |
| `Initialize(Vector<Single>,String)` | Initializes the context with model parameters. |
| `PrepareGradientsForUpdate(Vector<Half>,Vector<Single>)` | Converts FP16 gradients to FP32, unscales them, and checks for overflow. |
| `Reset` | Resets the context, clearing all weights and statistics. |
| `ToString` | Gets a summary of the context's current state. |
| `UpdateMasterWeights(Vector<Single>,Single,String)` | Updates master weights with FP32 gradients after unscaling. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_fp32SnapshotPool` | Per-tensor FP32 master snapshot pool — keyed by tensor reference identity so the SAME tensor across training steps reuses the same backing array. |
| `_masterWeights` | Master weights stored in FP32 for precise updates. |
| `_workingWeights` | Working weights stored in FP16 for fast computation. |

