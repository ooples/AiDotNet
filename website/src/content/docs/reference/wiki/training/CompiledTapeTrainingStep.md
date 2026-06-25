---
title: "CompiledTapeTrainingStep<T>"
description: "Compiled training step — auto-compiles the forward + backward pass on the first step, then replays the compiled plan on subsequent steps for near-zero overhead training."
section: "API Reference"
---

`Helpers & Utilities` · `AiDotNet.Training`

Compiled training step — auto-compiles the forward + backward pass on the first step,
then replays the compiled plan on subsequent steps for near-zero overhead training.

**How it works:**

- **Step 1 (tracing):** Enables GraphMode, traces the forward pass + loss computation

through the layer stack, compiles a CompiledTrainingPlan with backward pass, and executes it.

- **Steps 2+ (replay):** Calls plan.Step() which replays the compiled forward + backward

as flat delegate arrays with pre-allocated gradient buffers. Zero allocation, zero dispatch overhead.

**Recompilation triggers:**

- Input shape changes (different batch size, sequence length, etc.)
- Explicit Invalidate() call (model structure changed)
- Compilation failure (falls back to eager TapeTrainingStep for that shape)

## Methods

| Method | Summary |
|:-----|:--------|
| `CollectDeduplicatedParameters(IReadOnlyList<ITrainableLayer<>>)` | Deduplicates trainable parameter tensors by reference identity. |
| `GetFusedStepCount` | Gets the count of successful fused-step executions on the calling thread. |
| `GetLastFallbackException` | AiDotNet#1395: read the last exception that caused `Tensor{` to fall back, or `null` if the most recent fallback was due to one of the explicit return-false paths (plan switch, config drift, EnableCompilation=false, etc.) rather than a swal… |
| `InvalidateIfLayerSetChanged(IReadOnlyList<>)` | Invalidates the compiled plan cache. |
| `RememberLayerSet(IReadOnlyList<>)` | Captures the current trainable-layer set's reference identities so a subsequent call can detect a model switch. |
| `ResetFusedStepCount` | Resets the fused-step counter on the calling thread to zero. |
| `ShapesEqual(Int32[],Int32[])` | AiDotNet#1331 helper: structural shape equality. |
| `Step(IReadOnlyList<ITrainableLayer<>>,Tensor<>,Tensor<>,,Func<Tensor<>,Tensor<>>,Func<Tensor<>,Tensor<>,Tensor<>>)` | Executes a single compiled training step. |
| `StepMixedPrecision(IReadOnlyList<Tensor<Single>>,Tensor<Single>,Tensor<Single>,Single,Func<Tensor<Single>,Tensor<Single>>,Func<Tensor<Single>,Tensor<Single>,Tensor<Single>>,Int32[])` | One mixed-precision (FP16 activation storage) compiled training step. |
| `TryStepWithFusedOptimizer(IReadOnlyList<ITrainableLayer<>>,Tensor<>,Tensor<>,Func<Tensor<>,Tensor<>>,Func<Tensor<>,Tensor<>,Tensor<>>,OptimizerType,Single,Single,Single,Single,Single,,Double,LrSchedule,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Compiled training step with **fused optimizer** — forward + backward + parameter update all run in one compiled kernel. |
| `UpdateParametersSGD(Tensor<>[],Tensor<>[],,INumericOperations<>)` | In-place SGD: param[i] -= lr * grad[i] for each element. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_cachedLayerSetIdentities` | AiDotNet#1406: identity of the trainable-layer set that produced `_cachedParameters` and the cached compiled plan. |
| `_configuredOptimizerConfig` | Snapshot of the hyperparameters passed to `ICompiledTrainingPlan.ConfigureOptimizer` on `_configuredPlan`. |
| `_configuredPlan` | The single plan that has been configured with an optimizer on this thread. |
| `_fusedStepCount` | Counter of successful fused-step executions on this thread. |
| `_fusedUnavailableTypes` | Set once on the calling thread when an AMSGrad fused step fails because the linked Tensors build can't run the AMSGrad kernel. |
| `_lastFallbackException` | AiDotNet#1395: when `Tensor{` falls back via the catch path, the underlying exception is stored here so the caller (NeuralNetworkBase) can surface it in the "fused has committed but step N can't engage" InvalidOperationException. |
| `_persistentInput` | AiDotNet#1331: persistent input tensor reused across `Tensor{` calls. |
| `_persistentTarget` | AiDotNet#1331: persistent target tensor. |

