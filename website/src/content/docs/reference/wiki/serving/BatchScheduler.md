---
title: "BatchScheduler<T>"
description: "Schedules sequences for continuous batching based on priority and resource constraints."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Serving.ContinuousBatching`

Schedules sequences for continuous batching based on priority and resource constraints.

## For Beginners

The scheduler is like a restaurant host.

When new customers arrive (requests), the host must decide:

- Who gets seated next? (priority)
- How many can we serve at once? (batch size)
- Do we have enough tables/kitchen capacity? (memory/compute)
- Should we pause someone's meal to serve urgent customers? (preemption)

The scheduler makes these decisions to maximize throughput while
ensuring fairness and meeting priority requirements.

## How It Works

The scheduler determines which sequences should be processed in each iteration,
balancing priorities, fairness, and memory constraints.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `BatchScheduler` | Creates a new batch scheduler with default configuration. |
| `BatchScheduler(BatchSchedulerConfig)` | Creates a new batch scheduler with the specified configuration. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Config` | Gets the scheduler configuration. |
| `PreemptedCount` | Gets the number of preempted sequences waiting to resume. |
| `RunningCount` | Gets the number of sequences currently being processed. |
| `WaitingCount` | Gets the number of sequences currently waiting to be processed. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AddSequence(SequenceState<>)` | Adds a new sequence to the waiting queue. |
| `CancelSequence(Int64)` | Cancels a sequence, removing it from all queues. |
| `CompleteSequence(SequenceState<>)` | Marks a sequence as completed and removes it from the running set. |
| `GetRunningSequences` | Gets all currently running sequences. |
| `GetStatistics` | Gets statistics about the scheduler state. |
| `PreemptSequence(SequenceState<>)` | Preempts a running sequence, moving it to the preempted queue. |
| `ReorderByPriority` | Reorders running sequences by priority. |
| `ScheduleNextBatch` | Schedules the next batch of sequences to process. |

