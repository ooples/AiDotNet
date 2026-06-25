---
title: "NTMAlgorithm<T, TInput, TOutput>"
description: "Implementation of Neural Turing Machine (NTM) for meta-learning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.MetaLearning.Algorithms`

Implementation of Neural Turing Machine (NTM) for meta-learning.

## For Beginners

NTM is like a neural computer with RAM:

**How it works:**

1. Controller network processes inputs like a CPU
2. Generates read/write keys for memory access
3. Attention mechanism determines where to read/write
4. External memory stores information persistently
5. Differentiable operations allow end-to-end learning

**Key difference from standard NN:**

- Standard NN: Fixed computation graph
- NTM: Can learn to store and retrieve information dynamically
- Like giving a neural network a scratchpad to work with

## How It Works

Neural Turing Machines augment neural networks with an external memory matrix
and differentiable attention mechanisms for reading and writing. This enables
algorithms to be learned and executed within the neural network itself.

**Algorithm - Neural Turing Machine:**

**Key Insights:**

1. **Differentiable Memory**: Both reading and writing use differentiable

attention, allowing the entire system to be trained with backpropagation.

2. **Algorithmic Learning**: NTM can learn to implement algorithms like

sorting, copying, and associative recall directly from examples.

3. **Variable Computation**: The computation graph can change based on

what's stored in memory, enabling dynamic reasoning.

4. **Persistent State**: Information can be stored across timesteps,

enabling long-term memory and reasoning.

**Production Features:**

- LSTM or MLP controllers
- Multiple read/write heads
- Content-based and location-based addressing
- Memory initialization strategies
- Memory usage monitoring
- Differentiable memory operations

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `NTMAlgorithm(NTMOptions<,,>)` | Initializes a new instance of the NTMAlgorithm class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AlgorithmType` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Adapt(IMetaLearningTask<,,>)` |  |
| `AddMemoryRegularization()` | Adds memory regularization to the loss. |
| `CombineInputWithReadContents(Tensor<>)` | Combines input with previous read contents. |
| `ComputeControllerGradients(Vector<>,)` | Computes gradients for controller parameters using finite differences. |
| `ComputeCurrentLoss` | Computes current loss by running forward pass on cached inputs. |
| `GetOptions` |  |
| `InitializeMemory` | Initializes memory with default values. |
| `MetaTrain(TaskBatch<,,>)` |  |
| `ProcessSequence(,)` | Processes a sequence of inputs and targets. |
| `ProcessSupportSet(,,NTMModel<,,>)` | Processes support set to initialize memory state. |
| `ProcessTimestep(Tensor<>,Tensor<>)` | Processes a single timestep. |
| `ResetMemoryState` | Resets the memory state for a new episode. |
| `SetControllerParameters(Vector<>)` | Sets controller parameters (updates internal weights). |
| `TrainEpisode(IMetaLearningTask<,,>)` | Trains the NTM on a single episode. |
| `UpdateComponents()` | Updates all NTM components through backpropagation. |

