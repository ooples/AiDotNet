---
title: "PipelineParallelModel<T, TInput, TOutput>"
description: "Implements Pipeline Parallel model wrapper - splits model into stages across ranks."
section: "API Reference"
---

`Models & Types` · `AiDotNet.DistributedTraining`

Implements Pipeline Parallel model wrapper - splits model into stages across ranks.

## For Beginners

Pipeline parallelism is like an assembly line for training. Imagine a deep neural network as
a tall building - instead of one person (GPU) handling all floors, we assign different floors
to different people. Process 0 handles layers 0-10, Process 1 handles layers 11-20, etc.

To keep everyone busy (avoid idle time), we split each batch into smaller "micro-batches" that
flow through the pipeline like cars on an assembly line. While Process 1 is working on micro-batch 1,
Process 0 can start on micro-batch 2.

## How It Works

**Strategy Overview:**
Pipeline Parallelism divides the model vertically into stages, with each process
owning specific layers. Input mini-batches are divided into micro-batches that flow through
the pipeline stages sequentially. This enables training models too large to fit on a single device
while maintaining good hardware utilization through micro-batch pipelining.

**Supported Features (Issue #463):**

- **7 Pipeline Schedules**: GPipe, 1F1B, ZB-H1, ZB-H2, ZB-V, Interleaved 1F1B, Looped BFS.

Zero Bubble schedules decompose backward into BackwardInput + BackwardWeight for optimal throughput.

- **Virtual Stages**: Multi-stage schedules (Interleaved 1F1B, Looped BFS, ZB-V) assign

multiple non-contiguous model chunks per rank, reducing pipeline bubble by factor V.

- **Micro-Batch Slicing**: Input is automatically sliced into micro-batches that flow

through the pipeline independently.

- **Backward Decomposition**: If the wrapped model implements `IPipelineDecomposableModel`,

BackwardInput and BackwardWeight are truly decomposed. Otherwise, a compatible emulation is used.

- **Activation Checkpointing**: Trade compute for memory by recomputing activations from

checkpoints during the backward pass.

- **Load-Balanced Partitioning**: Balance compute across stages via dynamic programming.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PipelineParallelModel(IFullModel<,,>,IShardingConfiguration<>,Int32,IPipelinePartitionStrategy<>,IPipelineSchedule<>,ActivationCheckpointConfig)` | Creates a new Pipeline Parallel model. |

## Properties

| Property | Summary |
|:-----|:--------|
| `CheckpointConfig` | Gets the activation checkpoint configuration. |
| `EstimatedBubbleFraction` | Gets the estimated pipeline bubble fraction for the current configuration. |
| `PartitionStrategy` | Gets the partition strategy, or null if using uniform partitioning. |
| `Schedule` | Gets the pipeline schedule used by this model. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AccumulateGradients(Vector<>,Vector<>)` | Accumulates gradients across micro-batches. |
| `Clone` |  |
| `ComputeBackwardTag(Int32,Int32)` | Computes a unique communication tag for backward pass gradients. |
| `ComputeForwardTag(Int32,Int32)` | Computes a unique communication tag for forward pass activations. |
| `Deserialize(Byte[])` |  |
| `ExecuteForward(PipelineOperation,Dictionary<Int32,>,Dictionary<Int32,>,Dictionary<Int32,>,Int32)` | Executes a forward operation, handling virtual stage routing and activation checkpointing. |
| `FreeConsumedActivations(Int32,Dictionary<Int32,>,Dictionary<Int32,>)` | Frees activations after backward has consumed them to reduce memory usage. |
| `GetMicroBatchTarget(Int32,Dictionary<Int32,>,)` | Gets the target for a specific micro-batch. |
| `GetModelMetadata` |  |
| `GetOperationKey(Int32,Int32)` | Gets a unique key for a (microBatchIndex, virtualStageIndex) combination. |
| `GetStageInput(Dictionary<Int32,>,Int32,Int32,Dictionary<Int32,>)` | Gets the input for this stage, receiving from the previous global virtual stage. |
| `InitializeLayerAwareSharding(ILayeredModel<>,Vector<>)` | Performs layer-aware partitioning that respects layer boundaries and balances computational cost across pipeline stages. |
| `InitializeSharding` | Initializes pipeline parallelism by partitioning parameters into stages, including virtual stage partitions for multi-stage schedules. |
| `LoadModel(String)` |  |
| `OnBeforeInitializeSharding` | Called before InitializeSharding to set up derived class state. |
| `Predict()` |  |
| `ReceiveAndAccumulateDownstreamGradients(Vector<>,Int32,Int32)` | Receives gradients from the downstream (next) stage and accumulates them. |
| `RetrieveMicroBatchInput(Int32,Dictionary<Int32,>,Dictionary<Int32,>,PipelineOperation)` | Retrieves the input for a micro-batch from cache, checkpoint, or recomputes it. |
| `SaveModel(String)` |  |
| `SendActivationsForward(,Int32,Int32)` | Sends activations to the next stage in the global virtual pipeline. |
| `SendGradientsUpstream(Vector<>,Int32,Int32)` | Sends gradients to the upstream (previous) stage in the global virtual pipeline. |
| `Serialize` |  |
| `ShouldCheckpointActivation(Int32)` | Determines whether an activation should be checkpointed based on configuration. |
| `SliceInputIntoMicroBatches()` | Slices input into micro-batches by converting to a vector and dividing evenly. |
| `SliceTargetIntoMicroBatches()` | Slices target output into micro-batches by converting to a vector and dividing evenly. |
| `Train(,)` |  |
| `UpdateLocalShardFromFullParameters(Vector<>)` | Updates the local parameter shard from a full parameter vector, correctly handling non-contiguous virtual stage partitions for V>1 schedules. |
| `WithParameters(Vector<>)` |  |

