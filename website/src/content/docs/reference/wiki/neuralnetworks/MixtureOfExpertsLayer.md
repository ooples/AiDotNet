---
title: "MixtureOfExpertsLayer<T>"
description: "Implements a Mixture-of-Experts (MoE) layer that routes inputs through multiple expert networks."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers`

Implements a Mixture-of-Experts (MoE) layer that routes inputs through multiple expert networks.

## For Beginners

Think of a Mixture-of-Experts as a team of specialists working together.

How it works:

- You have multiple "experts" (specialized neural networks)
- A "router" (gating network) decides which experts should handle each input
- Each expert processes the input independently
- The final output is a weighted combination of the experts' outputs

Why use MoE:

- Scalability: Add more experts to increase model capacity without proportionally increasing computation
- Specialization: Different experts learn to handle different types of inputs
- Efficiency: Only activate the most relevant experts for each input (sparse MoE)

Real-world analogy:
Imagine you're running a hospital with specialists:

- A cardiologist (expert 1) handles heart problems
- A neurologist (expert 2) handles brain issues
- A pediatrician (expert 3) handles children's health
- A triage nurse (router) directs patients to the right specialist(s)

The router learns to send cardiac patients to the cardiologist, neurological cases to the
neurologist, etc. This is more efficient than having one doctor handle everything, and allows
each specialist to become highly skilled in their area.

## How It Works

A Mixture-of-Experts layer contains multiple expert networks and a gating/routing network.
For each input, the router determines how much weight to give each expert's output,
allowing the model to specialize different experts for different types of inputs.
This architecture enables models with very high capacity while remaining computationally efficient
by activating only a subset of parameters per input.

**Key Features:**

- Support for any number of experts
- Learned routing via a dense gating network
- Softmax routing: All experts contribute with learned weights
- Top-K routing: Only the top K experts are activated per input
- Load balancing: Optional auxiliary loss to encourage balanced expert usage

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MixtureOfExpertsLayer(List<ILayer<>>,ILayer<>,Int32[],Int32[],Int32,IActivationFunction<>,Boolean,)` | Initializes a new instance of the `MixtureOfExpertsLayer` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AuxiliaryLossWeight` | Gets or sets the weight for the auxiliary load balancing loss. |
| `NumExperts` | Gets the number of experts in this MoE layer. |
| `ParameterCount` | Gets the total number of trainable parameters in the layer. |
| `SupportsTraining` | Gets a value indicating whether this layer supports training. |
| `UseAuxiliaryLoss` | Gets or sets a value indicating whether to use the auxiliary load balancing loss. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplySoftmax(Tensor<>)` | Applies softmax to routing logits to produce normalized probability weights. |
| `ApplyTopK(Tensor<>,Int32)` | Applies Top-K selection to routing weights, keeping only the K highest weights per batch item. |
| `Clone` | Creates a deep copy of this MoE layer. |
| `CombineExpertOutputs(List<Tensor<>>,Tensor<>)` | Combines expert outputs using routing weights to produce the final output. |
| `CombineExpertOutputsGpuDense(DirectGpuTensorEngine,List<Tensor<>>,Tensor<>,Int32)` | Combines expert outputs using dense routing weights on GPU. |
| `CombineExpertOutputsGpuSparse(DirectGpuTensorEngine,List<Tensor<>>,Tensor<>,Int32[],Int32,Int32)` | Combines expert outputs using sparse routing weights (Top-K) on GPU. |
| `ComputeAuxiliaryLoss` | Computes the load balancing auxiliary loss based on expert usage from the last forward pass. |
| `ComputeRouterGradient(Tensor<>,List<Tensor<>>,Tensor<>,Tensor<>)` | Computes the gradient for the router during backpropagation. |
| `DivideByBroadcastGpuHelper(DirectGpuTensorEngine,Tensor<>,Tensor<>)` | Divides tensor A by broadcast of tensor B along axis 1. |
| `Forward(Tensor<>)` | Performs the forward pass through the MoE layer. |
| `ForwardGpu(Tensor<>[])` | Performs the forward pass on GPU tensors by routing through experts. |
| `GetAuxiliaryLossDiagnostics` | Gets diagnostic information about expert usage and load balancing. |
| `GetDiagnostics` | Gets diagnostic information about this component's state and behavior. |
| `GetParameterGradients` | Sets all trainable parameters from a single vector. |
| `GetParameters` | Gets all trainable parameters as a single vector. |
| `IsExpertUsedInBatch(Int32)` | Checks if a specific expert was used for any batch item. |
| `NormalizeOutputGradient(Tensor<>)` | Checks if a specific expert is active for a specific batch item in Top-K routing. |
| `OnFirstForward(Tensor<>)` |  |
| `ResetState` | Resets the internal state of the layer, clearing all cached values. |
| `UpdateParameters()` | Updates all trainable parameters using the specified learning rate. |
| `WeightGradientByRouting(Tensor<>,Tensor<>,Int32)` | Weights the output gradient by the routing weight for a specific expert. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_activeExpertsDuringBackward` | Tracks which experts were active during the most recent forward pass, so that only those experts have their parameters updated. |
| `_auxiliaryLossWeight` | The weight for the auxiliary load balancing loss. |
| `_experts` | The collection of expert networks. |
| `_lastExpertOutputs` | Cached expert outputs from the most recent forward pass. |
| `_lastInput` | Cached input from the most recent forward pass. |
| `_lastPreActivation` | Cached combined output before activation from the most recent forward pass. |
| `_lastRoutingLogits` | Cached routing logits (before softmax) from the most recent forward pass. |
| `_lastRoutingWeights` | Cached routing weights from the most recent forward pass. |
| `_lastTopKIndices` | Cached top-K indices for sparse routing from the most recent forward pass. |
| `_originalInputShape` | Stores the original input shape for any-rank tensor support. |
| `_router` | The router/gating network that determines how to weight each expert's output. |
| `_topK` | The number of top experts to activate for each input (0 means use all experts). |
| `_useAuxiliaryLoss` | Indicates whether to compute and use the auxiliary load balancing loss. |

