---
title: "Distributed Training"
description: "All 48 public types in the AiDotNet.distributedtraining namespace, organized by kind."
section: "API Reference"
---

**48** public types in this namespace, organized by kind.

## Models & Types (35)

| Type | Summary |
|:-----|:--------|
| [`AsyncSGDOptimizer<T, TInput, TOutput>`](/docs/reference/wiki/distributedtraining/asyncsgdoptimizer/) | Implements Asynchronous SGD optimizer - allows asynchronous parameter updates without strict barriers. |
| [`DDPModel<T, TInput, TOutput>`](/docs/reference/wiki/distributedtraining/ddpmodel/) | Implements DDP (Distributed Data Parallel) model wrapper for distributed training. |
| [`DDPOptimizer<T, TInput, TOutput>`](/docs/reference/wiki/distributedtraining/ddpoptimizer/) | Implements true DDP (Distributed Data Parallel) optimizer - industry-standard gradient averaging. |
| [`ElasticOptimizer<T, TInput, TOutput>`](/docs/reference/wiki/distributedtraining/elasticoptimizer/) | Implements Elastic optimizer - supports dynamic worker addition/removal during training. |
| [`FSDPModel<T, TInput, TOutput>`](/docs/reference/wiki/distributedtraining/fsdpmodel/) | Implements FSDP (Fully Sharded Data Parallel) model wrapper that shards parameters across multiple processes. |
| [`FSDPOptimizer<T, TInput, TOutput>`](/docs/reference/wiki/distributedtraining/fsdpoptimizer/) | Implements FSDP (Fully Sharded Data Parallel) optimizer wrapper that coordinates optimization across multiple processes. |
| [`GPipeSchedule<T>`](/docs/reference/wiki/distributedtraining/gpipeschedule/) | Implements the GPipe scheduling strategy: all forward passes first, then all backward passes. |
| [`GlooCommunicationBackend<T>`](/docs/reference/wiki/distributedtraining/gloocommunicationbackend/) | Gloo-based communication backend for CPU-based collective operations. |
| [`GradientCompressionOptimizer<T, TInput, TOutput>`](/docs/reference/wiki/distributedtraining/gradientcompressionoptimizer/) |  |
| [`HybridShardedModel<T, TInput, TOutput>`](/docs/reference/wiki/distributedtraining/hybridshardedmodel/) | Implements 3D Parallelism (Hybrid Sharded) model - combines data, tensor, and pipeline parallelism. |
| [`HybridShardedOptimizer<T, TInput, TOutput>`](/docs/reference/wiki/distributedtraining/hybridshardedoptimizer/) | Implements 3D Parallelism optimizer - coordinates across data, tensor, and pipeline dimensions. |
| [`InMemoryCommunicationBackend<T>`](/docs/reference/wiki/distributedtraining/inmemorycommunicationbackend/) | Provides an in-memory implementation of distributed communication for testing and single-machine scenarios. |
| [`Interleaved1F1BSchedule<T>`](/docs/reference/wiki/distributedtraining/interleaved1f1bschedule/) | Implements the Interleaved 1F1B pipeline schedule with multiple virtual stages per rank. |
| [`LoadBalancedPartitionStrategy<T>`](/docs/reference/wiki/distributedtraining/loadbalancedpartitionstrategy/) | Partitions model parameters across pipeline stages using estimated computational cost per layer. |
| [`LocalSGDOptimizer<T, TInput, TOutput>`](/docs/reference/wiki/distributedtraining/localsgdoptimizer/) | Implements Local SGD distributed training optimizer - parameter averaging after local optimization. |
| [`LoopedBFSSchedule<T>`](/docs/reference/wiki/distributedtraining/loopedbfsschedule/) | Implements the Looped BFS (Breadth-First Schedule) pipeline schedule with multiple virtual stages per rank. |
| [`MPICommunicationBackend<T>`](/docs/reference/wiki/distributedtraining/mpicommunicationbackend/) | MPI.NET-based communication backend for production distributed training. |
| [`NCCLCommunicationBackend<T>`](/docs/reference/wiki/distributedtraining/ncclcommunicationbackend/) | NVIDIA NCCL-based communication backend for GPU-to-GPU communication. |
| [`OneForwardOneBackwardSchedule<T>`](/docs/reference/wiki/distributedtraining/oneforwardonebackwardschedule/) | Implements the 1F1B (One-Forward-One-Backward) pipeline schedule. |
| [`ParameterAnalyzer<T>`](/docs/reference/wiki/distributedtraining/parameteranalyzer/) | Analyzes model parameters and creates optimized groupings for distributed communication. |
| [`ParameterGroup<T>`](/docs/reference/wiki/distributedtraining/parametergroup/) | Represents a group of parameters that should be communicated together. |
| [`PipelineParallelModel<T, TInput, TOutput>`](/docs/reference/wiki/distributedtraining/pipelineparallelmodel/) | Implements Pipeline Parallel model wrapper - splits model into stages across ranks. |
| [`PipelineParallelOptimizer<T, TInput, TOutput>`](/docs/reference/wiki/distributedtraining/pipelineparalleloptimizer/) | Implements Pipeline Parallel optimizer - coordinates optimization across pipeline stages. |
| [`TensorParallelModel<T, TInput, TOutput>`](/docs/reference/wiki/distributedtraining/tensorparallelmodel/) | Implements Tensor Parallel model wrapper - splits individual layers across ranks (Megatron-LM style). |
| [`TensorParallelOptimizer<T, TInput, TOutput>`](/docs/reference/wiki/distributedtraining/tensorparalleloptimizer/) | Implements Tensor Parallel optimizer - coordinates updates for tensor-parallel layers. |
| [`UniformPartitionStrategy<T>`](/docs/reference/wiki/distributedtraining/uniformpartitionstrategy/) | Divides model parameters evenly across pipeline stages. |
| [`ZeRO1Model<T, TInput, TOutput>`](/docs/reference/wiki/distributedtraining/zero1model/) | Implements ZeRO Stage 1 model wrapper - shards optimizer states only. |
| [`ZeRO1Optimizer<T, TInput, TOutput>`](/docs/reference/wiki/distributedtraining/zero1optimizer/) | Implements ZeRO Stage 1 optimizer - shards optimizer states only. |
| [`ZeRO2Model<T, TInput, TOutput>`](/docs/reference/wiki/distributedtraining/zero2model/) | Implements ZeRO Stage 2 model wrapper - shards optimizer states and gradients. |
| [`ZeRO2Optimizer<T, TInput, TOutput>`](/docs/reference/wiki/distributedtraining/zero2optimizer/) | Implements ZeRO Stage 2 optimizer - shards gradients and optimizer states across ranks. |
| [`ZeRO3Model<T, TInput, TOutput>`](/docs/reference/wiki/distributedtraining/zero3model/) | Implements ZeRO Stage 3 model wrapper - full sharding of parameters, gradients, and optimizer states. |
| [`ZeRO3Optimizer<T, TInput, TOutput>`](/docs/reference/wiki/distributedtraining/zero3optimizer/) | Implements ZeRO Stage 3 optimizer - full sharding equivalent to FSDP. |
| [`ZeroBubbleH1Schedule<T>`](/docs/reference/wiki/distributedtraining/zerobubbleh1schedule/) | Implements the Zero Bubble H1 (ZB-H1) pipeline schedule. |
| [`ZeroBubbleH2Schedule<T>`](/docs/reference/wiki/distributedtraining/zerobubbleh2schedule/) | Implements the Zero Bubble H2 (ZB-H2) pipeline schedule. |
| [`ZeroBubbleVSchedule<T>`](/docs/reference/wiki/distributedtraining/zerobubblevschedule/) | Implements the Zero Bubble V (ZB-V) pipeline schedule with 2 virtual stages per rank. |

## Base Classes (3)

| Type | Summary |
|:-----|:--------|
| [`CommunicationBackendBase<T>`](/docs/reference/wiki/distributedtraining/communicationbackendbase/) | Provides base implementation for distributed communication backends. |
| [`ShardedModelBase<T, TInput, TOutput>`](/docs/reference/wiki/distributedtraining/shardedmodelbase/) | Provides base implementation for distributed models with parameter sharding. |
| [`ShardedOptimizerBase<T, TInput, TOutput>`](/docs/reference/wiki/distributedtraining/shardedoptimizerbase/) | Provides base implementation for distributed optimizers with parameter sharding. |

## Interfaces (4)

| Type | Summary |
|:-----|:--------|
| [`ICommunicationBackend<T>`](/docs/reference/wiki/distributedtraining/icommunicationbackend/) | Defines the contract for distributed communication backends. |
| [`IShardedModel<T, TInput, TOutput>`](/docs/reference/wiki/distributedtraining/ishardedmodel/) | Defines the contract for models that support distributed training with parameter sharding. |
| [`IShardedOptimizer<T, TInput, TOutput>`](/docs/reference/wiki/distributedtraining/ishardedoptimizer/) | Defines the contract for optimizers that support distributed training with parameter sharding. |
| [`IShardingConfiguration<T>`](/docs/reference/wiki/distributedtraining/ishardingconfiguration/) | Configuration for parameter sharding in distributed training. |

## Enums (2)

| Type | Summary |
|:-----|:--------|
| [`RecomputeStrategy`](/docs/reference/wiki/distributedtraining/recomputestrategy/) | Strategy for recomputing activations during the backward pass. |
| [`ReductionOperation`](/docs/reference/wiki/distributedtraining/reductionoperation/) | Defines the supported reduction operations for collective communication. |

## Options & Configuration (2)

| Type | Summary |
|:-----|:--------|
| [`ActivationCheckpointConfig`](/docs/reference/wiki/distributedtraining/activationcheckpointconfig/) | Configuration for activation checkpointing in pipeline parallel training. |
| [`ShardingConfiguration<T>`](/docs/reference/wiki/distributedtraining/shardingconfiguration/) | Default implementation of sharding configuration for distributed training. |

## Helpers & Utilities (2)

| Type | Summary |
|:-----|:--------|
| [`CommunicationManager`](/docs/reference/wiki/distributedtraining/communicationmanager/) | Central manager for distributed communication operations. |
| [`DistributedExtensions`](/docs/reference/wiki/distributedtraining/distributedextensions/) | Provides extension methods for easily enabling distributed training on models and optimizers. |

