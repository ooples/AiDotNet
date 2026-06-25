---
title: "GradientBasedOptimizerBase<T, TInput, TOutput>"
description: "Represents a base class for gradient-based optimization algorithms."
section: "API Reference"
---

`Base Classes` · `AiDotNet.Optimizers`

Represents a base class for gradient-based optimization algorithms.

## For Beginners

Think of gradient-based optimization like finding the bottom of a valley:

- You start at a random point on a hilly landscape (your initial model parameters)
- You look around to see which way is steepest downhill (calculate the gradient)
- You take a step in that direction (update the parameters)
- You repeat this process until you reach the bottom of the valley (optimize the model)

This approach helps the model learn by gradually adjusting its parameters to minimize errors.

## How It Works

Gradient-based optimizers use the gradient of the loss function to update the model parameters
in a direction that minimizes the loss. This base class provides common functionality for
various gradient-based optimization techniques.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GradientBasedOptimizerBase(IFullModel<,,>,GradientBasedOptimizerOptions<,,>)` | Initializes a new instance of the GradientBasedOptimizerBase class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ActiveRegularization` | The active regularization applied during gradient updates. |
| `CurrentEpoch` | Gets the current training epoch. |
| `CurrentLossFunction` | The loss function currently used by the optimizer's gradient path. |
| `CurrentStep` | Gets the current training step (batch count). |
| `IsMixedPrecisionEnabled` | Gets whether mixed-precision training is enabled for this optimizer. |
| `LastComputedGradients` |  |
| `LearningRateScheduler` | Gets the current learning rate scheduler, if one is configured. |
| `SchedulerStepMode` | Gets the current scheduler step mode. |
| `SkipTrainingInEvaluation` | Gradient-based optimizers ALWAYS skip `model.Train()` during the pre-epoch `PrepareAndEvaluateSolution` call. |
| `SupportsGpuUpdate` | Gets whether this optimizer supports GPU-accelerated parameter updates. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyGradientClipping(Vector<>)` | Applies gradient clipping based on the configured options. |
| `ApplyGradients(Vector<>,IFullModel<,,>)` |  |
| `ApplyGradients(Vector<>,Vector<>,IFullModel<,,>)` |  |
| `ApplyGradients(Vector<>,Vector<>,IFullModel<,,>,Boolean)` | Applies gradients to parameters with explicit control over whether gradients are already unscaled. |
| `ApplyGradientsCore(Vector<>,Vector<>,IFullModel<,,>)` | Core implementation of gradient application without mixed-precision handling. |
| `ApplyGradientsWithMixedPrecision(Vector<>,Vector<>,IFullModel<,,>)` | Applies gradients with mixed-precision support (if enabled). |
| `ApplyMixedPrecisionScaling(Vector<>)` | Scales gradients for mixed-precision training if enabled. |
| `ApplyMomentum(Vector<>)` | Applies momentum to the gradient calculation. |
| `ApplyTapeGlobalNormGradientClipping(TapeStepContext<>,Double)` | Applies PyTorch-style global-norm gradient clipping across every gradient in the tape step's `Gradients` dictionary. |
| `ApplyTapeValueGradientClipping(TapeStepContext<>,Double)` | PyTorch `clip_grad_value_`-style element-wise clamp across every gradient in the tape step's `Gradients` dictionary: each element is bounded to `[-maxValue, +maxValue]`. |
| `AreGradientsExploding(Double)` | Checks if the current gradients are exhibiting exploding gradient behavior. |
| `AreGradientsVanishing(Double)` | Checks if the current gradients are exhibiting vanishing gradient behavior. |
| `BeginOptimizeRun(Object)` | Puts the model into training mode at the start of an Optimize run. |
| `CalculateGradient(IFullModel<,,>,,)` | Calculates the gradient for the given model and input data. |
| `CalculateGradient(IFullModel<,,>,,,Int32[])` | Calculates the gradient for a given solution using a batch of training data. |
| `ComputeHessianEfficiently(IFullModel<,,>,OptimizationInputData<,,>)` | Computes the Hessian matrix (second derivatives) more efficiently when the model supports explicit gradient computation. |
| `ComputeHessianFiniteDifferences(IFullModel<,,>,OptimizationInputData<,,>)` | Computes the Hessian matrix using traditional finite differences (fallback method). |
| `ComputeParameterFingerprint(IFullModel<,,>)` | Computes a fast fingerprint of the model's current parameter state for gradient cache invalidation. |
| `CreateBatcher(OptimizationInputData<,,>,Int32)` | Creates a data batcher for the given optimization input data using configured sampling options. |
| `CreateBatcher(OptimizationInputData<,,>,Int32,IDataSampler)` | Creates a data batcher with a custom sampler, overriding the configured options. |
| `CreateRegularization(GradientDescentOptimizerOptions<,,>)` | Creates a regularization technique based on the provided options. |
| `DisableMixedPrecision` | Disables mixed-precision training and releases associated resources. |
| `DisposeGpuState` | Disposes GPU-allocated optimizer state. |
| `EnableMixedPrecision(MixedPrecisionConfig)` | Enables mixed-precision training for this optimizer. |
| `EndOptimizeRun(Object)` | Returns the model to eval mode at the end of an Optimize run. |
| `GenerateGradientCacheKey(IFullModel<,,>,,)` | Generates a unique key for caching gradients. |
| `GetCurrentLearningRate` | Gets the current learning rate being used by this optimizer. |
| `GetCurrentLearningRateAsT` | Gets the current learning rate as the generic type T. |
| `GetGradientNorm` | Gets the L2 norm of the last computed gradients. |
| `GetMixedPrecisionContext` | Gets the mixed-precision training context (if enabled). |
| `GetUnscaledGradients` | Gets the last computed gradients, unscaled from loss scaling if mixed precision is enabled. |
| `HasAnomalousTapeGradients(TapeStepContext<>)` | PyTorch GradScaler-style anomaly probe across every gradient in the tape step's `Gradients` dictionary: returns `true` if any element is NaN or ±Inf. |
| `InitializeGpuState(Int32,IDirectGpuBackend)` | Initializes optimizer state on the GPU for a given parameter count. |
| `IsConvergedAgainstPreviousEpoch(Int32,OptimizationStepData<,,>,OptimizationStepData<,,>,Double)` | Returns true when the per-epoch convergence signal is below `tolerance`. |
| `IsInWarmupPhase` | Determines whether the scheduler is currently in the warmup phase. |
| `LineSearch(IFullModel<,,>,Vector<>,Vector<>,OptimizationInputData<,,>)` | Performs a line search to find an appropriate step size. |
| `NotifyEpochStart(Int32)` | Notifies the sampler that a new epoch has started (for epoch-aware samplers). |
| `OnBatchEnd` | Called at the end of each training batch to update scheduler state if applicable. |
| `OnEpochEnd` | Called at the end of each training epoch to update scheduler state if applicable. |
| `OnInitialTrainingCompleted` | No-op for gradient-based optimizers — `SkipTrainingInEvaluation` is already `true` from the very first call. |
| `OnModelChanged(IFullModel<,,>,IFullModel<,,>)` | When the optimizer's model is (re-)set and the caller did NOT explicitly configure `LossFunction`, adopt the model's `DefaultLossFunction`. |
| `ParameterBitsToLong()` | Converts a numeric parameter value into a stable 64-bit bit pattern for fingerprint mixing. |
| `Reset` | Resets the optimizer to its initial state. |
| `ResolveParameterChunks(IParameterizable<,,>)` | Resolves the per-layer parameter chunks for an arbitrary `IParameterizable`. |
| `ReverseUpdate(Vector<>,Vector<>)` | Reverses a gradient update to recover original parameters. |
| `SetLearningRate(Double)` | Sets the current learning rate, synchronizing both the double field (_currentLearningRate) and the generic T field (CurrentLearningRate) used by derived optimizers. |
| `SetRegularization(IRegularization<,,>)` | Replaces the active regularization on this optimizer at runtime. |
| `Step(TapeStepContext<>)` |  |
| `StepScheduler` | Steps the learning rate scheduler and updates the current learning rate. |
| `SynthesizeTapeStepContext(INeuralNetwork<>,Vector<>)` | #1413 helper: build a `TapeStepContext` from a model's live parameter chunks plus a flat gradient vector. |
| `TryGetFusedLrSchedule(LrSchedule)` | Resolves this optimizer's attached LR scheduler (if any) into a fused-side `LrSchedule` for `IFusedOptimizerSpec` implementations. |
| `UpdateOptions(OptimizationAlgorithmOptions<,,>)` | Updates the options for the gradient-based optimizer. |
| `UpdateParameters(List<ILayer<>>)` | Updates the parameters of the model based on the calculated gradients. |
| `UpdateParameters(Matrix<>,Matrix<>)` | Updates a matrix of parameters based on the calculated gradient. |
| `UpdateParameters(Tensor<>,Tensor<>)` | Updates a tensor of parameters based on the calculated gradient. |
| `UpdateParameters(Vector<>,Vector<>)` | Updates a vector of parameters based on the calculated gradient. |
| `UpdateParametersGpu(IGpuBuffer,IGpuBuffer,Int32,IDirectGpuBackend)` | Updates parameters on the GPU using optimizer-specific GPU kernels. |
| `UpdateSolution(IFullModel<,,>,Vector<>)` | Updates the current solution based on the calculated gradient. |

## Fields

| Field | Summary |
|:-----|:--------|
| `GradientCache` | A cache for storing and retrieving gradients to improve performance. |
| `GradientOptions` | Options specific to gradient-based optimization algorithms. |
| `LossFunction` | A method used to compare the predicted values vs the actual values. |
| `Regularization` | A method used to regularize the parameters so they don't get out of control. |
| `_currentEpoch` | The current epoch number for scheduler tracking. |
| `_currentLearningRate` | The current learning rate used in the optimization process. |
| `_currentMomentum` | The current momentum factor used in the optimization process. |
| `_currentStep` | The current step (batch) number for scheduler tracking. |
| `_gpuState` | GPU-resident optimizer state. |
| `_gpuStateInitialized` | Whether GPU state has been initialized. |
| `_lastComputedGradients` | The gradients computed during the last optimization step. |
| `_learningRateScheduler` | The learning rate scheduler to use for adjusting learning rate during training. |
| `_mixedPrecisionContext` | Mixed-precision training context (null if mixed-precision is disabled). |
| `_previousGradient` | The gradient from the previous optimization step, used for momentum calculations. |
| `_schedulerStepMode` | Specifies when to step the learning rate scheduler. |

