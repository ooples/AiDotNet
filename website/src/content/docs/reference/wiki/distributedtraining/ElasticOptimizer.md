---
title: "ElasticOptimizer<T, TInput, TOutput>"
description: "Implements Elastic optimizer - supports dynamic worker addition/removal during training."
section: "API Reference"
---

`Models & Types` · `AiDotNet.DistributedTraining`

Implements Elastic optimizer - supports dynamic worker addition/removal during training.

## For Beginners

Elastic training is like having a flexible team size. Workers can join or leave during
training without stopping everything:

Scenario 1 - Fault tolerance:

- Start with 8 GPUs training your model
- GPU 3 fails → automatically detected
- Training continues with 7 GPUs (parameters redistributed)
- New GPU joins → training scales back to 8 GPUs

Scenario 2 - Cloud cost optimization:

- Use cheap "spot instances" that can be taken away anytime
- When instance is preempted, training continues with remaining workers
- New instance joins when available

This is critical for long training jobs where failures are expected.

## How It Works

**Strategy Overview:**
Elastic training (TorchElastic, Horovod Elastic) enables dynamic scaling of workers during
training. Workers can be added or removed without stopping the training job, supporting:

- Fault tolerance: Replace failed workers automatically
- Auto-scaling: Add workers during peak hours, remove during off-peak
- Spot instance usage: Tolerate preemptions, use cheaper compute

When world size changes, the optimizer handles re-sharding parameters and optimizer states
across the new worker set. This requires checkpointing and careful state management.

**Use Cases:**

- Long training jobs (days/weeks) where failures will occur
- Cloud training with spot/preemptible instances (save 60-90% cost)
- Auto-scaling based on load or time of day
- Fault tolerance for production training pipelines

**Trade-offs:**

- Memory: Must handle dynamic re-sharding
- Communication: Overhead during worker changes (re-sharding, sync)
- Complexity: Very High - requires membership management, state re-distribution
- Convergence: Learning rate scheduling must account for dynamic world size
- Best for: Long jobs, cost-sensitive scenarios, production ML pipelines
- Limitation: Worker changes create temporary slowdown during re-sharding

**Implementation Note:**
This framework provides elastic optimizer infrastructure. Full production deployment
requires:

1. Membership/discovery service (etcd, ZooKeeper, or cloud-native)
2. Automatic checkpointing before worker changes
3. State re-sharding algorithms
4. Rendezvous mechanism for worker coordination

This implementation demonstrates the elastic pattern.

Example:

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ElasticOptimizer(IOptimizer<,,>,IShardingConfiguration<>,Int32,Int32)` | Creates an elastic optimizer. |

## Properties

| Property | Summary |
|:-----|:--------|
| `CanScaleDown` | Gets whether the optimizer can tolerate losing workers. |
| `CanScaleUp` | Gets whether the optimizer can accept more workers. |
| `CurrentWorkers` | Gets the current number of active workers. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Deserialize(Byte[])` |  |
| `DetectWorldSizeChange` | Detects if the world size has changed. |
| `HandleWorkerChange` | Handles worker addition or removal. |
| `Optimize(OptimizationInputData<,,>)` |  |
| `Serialize` |  |
| `SynchronizeOptimizerState` |  |

