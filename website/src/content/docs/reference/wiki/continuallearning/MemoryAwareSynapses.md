---
title: "MemoryAwareSynapses<T>"
description: "Implements Memory Aware Synapses (MAS) for continual learning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ContinualLearning`

Implements Memory Aware Synapses (MAS) for continual learning.

## For Beginners

MAS is similar to EWC but estimates weight importance in an
unsupervised way using the sensitivity of the network output to each weight. This means
it doesn't need task labels to compute importance.

## How It Works

**How it works:**

**Key Formula:**

Ω_i = 1/N × Σ_n |∂F(x_n)/∂θ_i|

where F is the network output and θ_i is weight i.

**Advantages over EWC:**

**Reference:** Aljundi, R., Babiloni, F., Elhoseiny, M., Rohrbach, M., and Tuytelaars, T.
"Memory Aware Synapses: Learning what (not) to forget" (2018). ECCV.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MemoryAwareSynapses(Double)` | Initializes a new instance of the MemoryAwareSynapses class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AccumulatesAcrossTasks` |  |
| `TaskCount` | Gets the number of tasks processed. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AfterTask(INeuralNetwork<>,ValueTuple<Tensor<>,Tensor<>>,Int32)` |  |
| `BeforeTask(INeuralNetwork<>,Int32)` |  |
| `ComputeLoss(INeuralNetwork<>)` |  |
| `ComputeOutputNormGradient(Tensor<>)` | Computes the gradient of the output L2 norm. |
| `ExtractSample(Tensor<>,Int32)` | Extracts a single sample from a batch tensor. |
| `ModifyGradients(INeuralNetwork<>,Vector<>)` |  |
| `Reset` |  |

