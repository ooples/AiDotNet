---
title: "MANNOptions<T, TInput, TOutput>"
description: "Configuration options for Memory-Augmented Neural Networks (MANN) algorithm."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.MetaLearning.Options`

Configuration options for Memory-Augmented Neural Networks (MANN) algorithm.

## For Beginners

MANN is like a neural network with a notebook:

1. The "controller" (neural network) processes inputs
2. The "memory" stores important information for later
3. Read heads retrieve relevant memories
4. Write heads store new information

This allows one-shot learning - see an example once, store it, use it later!

## How It Works

Memory-Augmented Neural Networks combine a neural network controller with an external
memory matrix. The network can read from and write to this memory, enabling rapid
learning by storing new information directly in memory during adaptation.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MANNOptions(IFullModel<,,>)` | Initializes a new instance of the MANNOptions class with the required meta-model. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AdaptationSteps` | Gets or sets the number of adaptation steps. |
| `CheckpointFrequency` | Gets or sets how often to save checkpoints. |
| `ClearMemoryBetweenTasks` | Gets or sets whether to clear memory between tasks. |
| `DataLoader` | Gets or sets the episodic data loader for sampling tasks. |
| `EnableCheckpointing` | Gets or sets whether to save model checkpoints. |
| `EvaluationFrequency` | Gets or sets how often to evaluate the meta-learner. |
| `EvaluationTasks` | Gets or sets the number of tasks to use for evaluation. |
| `GradientClipThreshold` | Gets or sets the maximum gradient norm for gradient clipping. |
| `InnerLearningRate` | Gets or sets the learning rate for the inner loop (not used in MANN). |
| `LossFunction` | Gets or sets the loss function for training. |
| `MemoryKeySize` | Gets or sets the dimension of memory keys. |
| `MemoryRegularization` | Gets or sets the memory regularization strength. |
| `MemoryRetentionRatio` | Gets or sets the ratio of memory to retain when clearing. |
| `MemorySize` | Gets or sets the number of memory slots. |
| `MemoryUsageThreshold` | Gets or sets the threshold for memory usage when consolidating. |
| `MemoryValueSize` | Gets or sets the dimension of memory values. |
| `MetaBatchSize` | Gets or sets the number of tasks to sample per meta-training iteration. |
| `MetaModel` | Gets or sets the meta-model (controller network) to be trained. |
| `MetaOptimizer` | Gets or sets the optimizer for network updates. |
| `NumClasses` | Gets or sets the number of output classes. |
| `NumMetaIterations` | Gets or sets the total number of meta-training iterations. |
| `NumReadHeads` | Gets or sets the number of read heads. |
| `NumWriteHeads` | Gets or sets the number of write heads. |
| `OuterLearningRate` | Gets or sets the learning rate for the outer loop (controller training). |
| `RandomSeed` | Gets or sets the random seed for reproducibility. |
| `ReadHeadOptions` | Gets or sets options for the read head configuration. |
| `UseCommonPatternsInitialization` | Gets or sets whether to initialize with common patterns. |
| `UseFirstOrder` | Gets or sets whether to use first-order approximation. |
| `UseHierarchicalMemory` | Gets or sets whether to use hierarchical memory organization. |
| `UseMemoryConsolidation` | Gets or sets whether to use memory consolidation. |
| `UseMemoryPreInitialization` | Gets or sets whether to pre-initialize memory. |
| `UseOutputSoftmax` | Gets or sets whether to apply softmax to output. |
| `UseValueProjection` | Gets or sets whether to project values before storing. |
| `WriteHeadOptions` | Gets or sets options for the write head configuration. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clone` | Creates a deep copy of the MANN options. |
| `IsValid` | Validates that all MANN configuration options are properly set. |

