---
title: "MemoryAwareSynapses<T, TInput, TOutput>"
description: "Memory Aware Synapses (MAS) strategy for continual learning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ContinualLearning.Strategies`

Memory Aware Synapses (MAS) strategy for continual learning.

## For Beginners

MAS estimates weight importance in an unsupervised way
by measuring how sensitive the network output is to each parameter. This means
it doesn't need task labels to compute importance - just unlabeled data!

## How It Works

**Key Insight:** If changing a parameter causes a large change in the
network output, that parameter is important and should be protected. This is
measured by the gradient of the output norm with respect to each parameter.

**How it works:**

**The Math:**

For each sample x_n:

1. Forward pass: y = F(x_n)

2. Compute output norm gradient: ∂||y||²/∂θ = 2y × (∂y/∂θ)

3. Importance: Ω_i = (1/N) × Σ_n |∂||y||²/∂θ_i|

**Advantages over EWC:**

**Reference:** Aljundi, R., Babiloni, F., Elhoseiny, M., Rohrbach, M., and Tuytelaars, T.
"Memory Aware Synapses: Learning what (not) to forget" (2018). ECCV.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MemoryAwareSynapses(ILossFunction<>,)` | Initializes a new MAS strategy with a lambda value. |
| `MemoryAwareSynapses(ILossFunction<>,MASOptions<>)` | Initializes a new MAS strategy with custom options. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ConsolidatedImportance` | Gets the consolidated importance values. |
| `Lambda` | Gets the regularization strength. |
| `MemoryUsageBytes` |  |
| `ModifiesArchitecture` |  |
| `Name` |  |
| `OptimalParameters` | Gets the optimal parameters from the last completed task. |
| `RequiresMemoryBuffer` |  |
| `TaskImportanceHistory` | Gets the importance history for all tasks. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AccumulateImportance(Vector<>)` | Accumulates task importance into the consolidated omega. |
| `AdjustGradients(Vector<>)` |  |
| `ComputeFiniteDifferenceFisher(IFullModel<,,>,,Vector<>)` | Computes Fisher diagonal using finite differences for non-gradient models. |
| `ComputeFiniteDifferenceImportance(IFullModel<,,>,,Vector<>)` | Computes importance using finite difference when gradients aren't available. |
| `ComputeFisherDiagonalImportance(IFullModel<,,>)` | Computes importance using Fisher diagonal (hybrid with EWC). |
| `ComputeHebbianImportance(IFullModel<,,>)` | Computes importance using Hebbian-style activation magnitudes. |
| `ComputeImportance(IFullModel<,,>)` | Computes parameter importance based on output sensitivity. |
| `ComputeMax(Vector<>)` | Computes the maximum value in a vector. |
| `ComputeMean(Vector<>)` | Computes the mean of a vector. |
| `ComputeOutputNorm(Vector<>)` | Computes the norm of an output vector. |
| `ComputeOutputNormGradient(Vector<>)` | Computes the gradient of output L2 norm. |
| `ComputeOutputSensitivity(IFullModel<,,>)` | Computes importance using output sensitivity (original MAS method). |
| `ComputeRandomProjectionImportance(IFullModel<,,>)` | Computes importance using random projection method. |
| `ComputeRegularizationLoss(IFullModel<,,>)` |  |
| `ConvertToVector()` | Converts output to a vector for gradient computation. |
| `CountNonZero(Vector<>)` | Counts non-zero elements in a vector. |
| `FinalizeTask(IFullModel<,,>)` |  |
| `GetRandomIndices(Int32,Int32)` | Gets random indices for sampling. |
| `GetStateForSerialization` |  |
| `NormalizeImportanceValues(Vector<>)` | Normalizes importance values. |
| `PrepareForTask(IFullModel<,,>,IDataset<,,>)` |  |
| `ProcessImportanceBatch(IFullModel<,,>,List<>,Vector<>)` | Processes a batch of samples for importance computation. |
| `ProcessSingleSampleImportance(IFullModel<,,>,,Vector<>)` | Processes a single sample for importance computation. |
| `Reset` |  |

