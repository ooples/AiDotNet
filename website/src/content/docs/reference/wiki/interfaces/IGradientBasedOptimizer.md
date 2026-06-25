---
title: "IGradientBasedOptimizer<T, TInput, TOutput>"
description: "IGradientBasedOptimizer<T, TInput, TOutput> — Interfaces in AiDotNet.Interfaces."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

_No summary documentation available yet._

## Properties

| Property | Summary |
|:-----|:--------|
| `LastComputedGradients` | Gets the gradients computed during the last optimization step. |
| `SupportsGpuUpdate` | Gets whether this optimizer supports GPU-accelerated parameter updates. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyGradients(Vector<>,IFullModel<,,>)` | Applies pre-computed gradients to a model's parameters. |
| `ApplyGradients(Vector<>,Vector<>,IFullModel<,,>)` | Applies pre-computed gradients to explicit original parameters (double-step safe). |
| `DisposeGpuState` | Disposes GPU-allocated optimizer state. |
| `GetCurrentLearningRate` | Gets the optimizer's current learning rate, which may have been adjusted by a scheduler. |
| `InitializeGpuState(Int32,IDirectGpuBackend)` | Initializes optimizer state on the GPU for a given parameter count. |
| `ReverseUpdate(Vector<>,Vector<>)` | Reverses a gradient update to recover original parameters before the update was applied. |
| `Step(TapeStepContext<>)` | Performs a parameter update step using the optimizer's update rule and the provided training context. |
| `UpdateParameters(List<ILayer<>>)` | Updates the parameters of all layers in a model based on their calculated gradients. |
| `UpdateParameters(Matrix<>,Matrix<>)` | Updates matrix parameters based on their gradients. |
| `UpdateParameters(Vector<>,Vector<>)` | Updates parameters based on their gradients. |
| `UpdateParametersGpu(IGpuBuffer,IGpuBuffer,Int32,IDirectGpuBackend)` | Updates parameters on the GPU using optimizer-specific GPU kernels. |

