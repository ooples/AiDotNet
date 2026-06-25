---
title: "NTMOptions<T, TInput, TOutput>"
description: "Configuration options for Neural Turing Machine (NTM) algorithm."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.MetaLearning.Options`

Configuration options for Neural Turing Machine (NTM) algorithm.

## For Beginners

NTM is like a neural computer with RAM:

1. Controller network processes inputs like a CPU
2. Generates read/write keys for memory access
3. Attention mechanism determines where to read/write
4. External memory stores information persistently
5. Differentiable operations allow end-to-end learning

This allows learning algorithms like sorting, copying, and associative recall!

## How It Works

Neural Turing Machines augment neural networks with an external memory matrix
and differentiable attention mechanisms for reading and writing. This enables
algorithms to be learned and executed within the neural network itself.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `NTMOptions(IFullModel<,,>)` | Initializes a new instance of the NTMOptions class with the required meta-model. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AdaptationSteps` | Gets or sets the number of adaptation steps. |
| `CheckpointFrequency` | Gets or sets how often to save checkpoints. |
| `ControllerHiddenSize` | Gets or sets the hidden size of the controller. |
| `ControllerType` | Gets or sets the type of controller to use. |
| `DataLoader` | Gets or sets the episodic data loader for sampling tasks. |
| `EnableCheckpointing` | Gets or sets whether to save model checkpoints. |
| `EvaluationFrequency` | Gets or sets how often to evaluate the meta-learner. |
| `EvaluationTasks` | Gets or sets the number of tasks to use for evaluation. |
| `GradientClipThreshold` | Gets or sets the maximum gradient norm for gradient clipping. |
| `InitializeMemory` | Gets or sets whether to initialize memory at start of episodes. |
| `InnerLearningRate` | Gets or sets the learning rate for the inner loop (not used in NTM). |
| `InnerOptimizer` | Gets or sets the optimizer for inner loop updates. |
| `LossFunction` | Gets or sets the loss function for training. |
| `MemoryInitialization` | Gets or sets how to initialize memory. |
| `MemorySharpnessRegularization` | Gets or sets the memory sharpness regularization strength. |
| `MemorySize` | Gets or sets the number of memory slots. |
| `MemoryUsageRegularization` | Gets or sets the memory usage regularization strength. |
| `MemoryWidth` | Gets or sets the dimension of each memory slot. |
| `MetaBatchSize` | Gets or sets the number of tasks to sample per meta-training iteration. |
| `MetaModel` | Gets or sets the meta-model (controller network) to be trained. |
| `MetaOptimizer` | Gets or sets the optimizer for network updates. |
| `NumClasses` | Gets or sets the number of output classes. |
| `NumMetaIterations` | Gets or sets the total number of meta-training iterations. |
| `NumReadHeads` | Gets or sets the number of read heads. |
| `NumWriteHeads` | Gets or sets the number of write heads. |
| `OuterLearningRate` | Gets or sets the learning rate for the outer loop (controller training). |
| `RandomSeed` | Gets or sets the random seed for reproducibility. |
| `UseFirstOrder` | Gets or sets whether to use first-order approximation. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clone` | Creates a deep copy of the NTM options. |
| `IsValid` | Validates that all NTM configuration options are properly set. |

