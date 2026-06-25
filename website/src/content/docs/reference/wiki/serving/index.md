---
title: "Serving"
description: "All 13 public types in the AiDotNet.serving namespace, organized by kind."
section: "API Reference"
---

**13** public types in this namespace, organized by kind.

## Models & Types (8)

| Type | Summary |
|:-----|:--------|
| [`BatchScheduler<T>`](/docs/reference/wiki/serving/batchscheduler/) | Schedules sequences for continuous batching based on priority and resource constraints. |
| [`BatcherStatistics`](/docs/reference/wiki/serving/batcherstatistics/) | Statistics about the batcher's operation. |
| [`GenerationRequest<T>`](/docs/reference/wiki/serving/generationrequest/) | Represents a request for text generation. |
| [`GenerationResult<T>`](/docs/reference/wiki/serving/generationresult/) | Result of a generation request. |
| [`SchedulerStatistics`](/docs/reference/wiki/serving/schedulerstatistics/) | Statistics about the scheduler state. |
| [`SequenceCompletedEventArgs<T>`](/docs/reference/wiki/serving/sequencecompletedeventargs/) | Event args for sequence completion. |
| [`SequenceState<T>`](/docs/reference/wiki/serving/sequencestate/) | Represents the state of a single sequence being processed in continuous batching. |
| [`TokenGeneratedEventArgs<T>`](/docs/reference/wiki/serving/tokengeneratedeventargs/) | Event args for token generation. |

## Enums (3)

| Type | Summary |
|:-----|:--------|
| [`SchedulingPolicy`](/docs/reference/wiki/serving/schedulingpolicy/) | Scheduling policies for batch scheduling. |
| [`SequenceStatus`](/docs/reference/wiki/serving/sequencestatus/) | Status of a sequence in the continuous batching system. |
| [`StopReason`](/docs/reference/wiki/serving/stopreason/) | Reasons why generation stopped. |

## Options & Configuration (2)

| Type | Summary |
|:-----|:--------|
| [`BatchSchedulerConfig`](/docs/reference/wiki/serving/batchschedulerconfig/) | Configuration for the batch scheduler. |
| [`ContinuousBatcherConfig`](/docs/reference/wiki/serving/continuousbatcherconfig/) | Configuration for the continuous batcher. |

