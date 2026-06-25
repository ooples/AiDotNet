---
title: "FedMeZO<T>"
description: "Implements FedMeZO — Memory-efficient Zeroth-Order optimization for federated LLM fine-tuning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.Adapters`

Implements FedMeZO — Memory-efficient Zeroth-Order optimization for federated LLM fine-tuning.

## For Beginners

Training large language models normally requires storing gradients
for all parameters, which takes enormous memory (often 3-4x the model size). Zeroth-order (ZO)
optimization estimates gradients by evaluating the loss at two slightly perturbed points —
no backpropagation needed. This reduces memory to just the model size plus a random seed.
FedMeZO brings this to federated learning: clients only need to share the scalar loss
difference and the random seed, making it extremely communication-efficient.

## How It Works

ZO gradient estimate (SPSA — Simultaneous Perturbation Stochastic Approximation):

Communication per client: {loss_diff: double, seed: int} — just 12 bytes instead of
the full parameter vector (millions to billions of doubles).

Multi-query ZO: For better gradient estimates, multiple perturbation directions can be
sampled per step, averaging the estimates.

Reference: Malladi, S., et al. (2024). "Fine-Tuning Language Models with Just Forward
Passes." NeurIPS 2023. FedMeZO extension for federated settings (2024).

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FedMeZO(Int32,Double,Double,Int32)` | Creates a new FedMeZO strategy. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AdapterParameterCount` |  |
| `CompressionRatio` |  |
| `LearningRate` | Gets the server learning rate. |
| `NumPerturbations` | Gets the number of perturbation directions per step. |
| `PerturbationScale` | Gets the ZO perturbation scale (epsilon). |

## Methods

| Method | Summary |
|:-----|:--------|
| `AggregateAdapters(Dictionary<Int32,Vector<>>,Dictionary<Int32,Double>)` |  |
| `AggregateFromMessages(Dictionary<Int32,ZOClientMessage[]>,Dictionary<Int32,Double>)` | Aggregates ZO gradient estimates from multiple client messages on the server. |
| `ComputePerturbedWeights(Vector<>,Int32)` | Computes the perturbed parameter vectors w+ = w + epsilon*z and w- = w - epsilon*z. |
| `CreateClientMessage(Double,Double,Int32)` | Creates the minimal message a client sends to the server in the FedMeZO protocol. |
| `EstimateGradient(Double,Double,Int32)` | Estimates the gradient from a single perturbation direction using the loss difference. |
| `EstimateGradientMultiQuery(ValueTuple<Double,Double,Int32>[])` | Estimates the gradient using multiple perturbation directions for better accuracy. |
| `ExtractAdapterParameters(Vector<>)` |  |
| `GeneratePerturbation(Int32)` | Generates a random perturbation vector z ~ N(0, I) from a seed. |
| `MergeAdapterParameters(Vector<>,Vector<>)` |  |
| `ReconstructGradientFromMessage(ZOClientMessage)` | Reconstructs the gradient estimate on the server from a client's compact message. |

