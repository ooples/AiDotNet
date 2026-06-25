---
title: "SequenceState<T>"
description: "Represents the state of a single sequence being processed in continuous batching."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Serving.ContinuousBatching`

Represents the state of a single sequence being processed in continuous batching.

## For Beginners

Think of this as tracking one person's order in a restaurant.

Traditional batching: Everyone orders at once, waits together, gets food together.
Continuous batching: People can order anytime, food comes when ready, new orders join ongoing batch.

SequenceState tracks:

- What tokens have been generated so far
- When this request started
- Whether generation is complete
- How many tokens are left to generate

## How It Works

Each sequence tracks its own progress through generation, including tokens generated,
KV-cache state, and completion status. This enables sequences to be added to and
removed from batches dynamically.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SequenceState(GenerationRequest<>)` | Creates a new sequence state from a generation request. |

## Properties

| Property | Summary |
|:-----|:--------|
| `BatchIndex` | Index in the current batch (-1 if not in batch). |
| `CacheSlot` | Cache slot index for this sequence. |
| `CompletedAt` | Timestamp when generation completed. |
| `CreatedAt` | Timestamp when the sequence was created. |
| `CumulativeLogProb` | Cumulative log probability of generated tokens. |
| `FinishReason` | Stop reason if generation is complete. |
| `GeneratedLength` | Number of tokens generated (excluding prompt). |
| `GenerationStartedAt` | Timestamp when generation started (after prefill). |
| `GenerationTime` | Gets the total generation time (after prefill). |
| `MaxNewTokens` | Maximum number of new tokens to generate. |
| `PrefillComplete` | Whether the prefill phase is complete. |
| `Priority` | Priority for scheduling (higher = more important). |
| `PromptLength` | Number of tokens from the original prompt. |
| `QueueTime` | Gets the time spent in queue (before generation started). |
| `Request` | The original request that created this sequence. |
| `SequenceId` | Unique identifier for this sequence. |
| `Status` | Current status of this sequence. |
| `TokenIds` | List of token IDs generated so far (including prompt tokens). |
| `TokensPerSecond` | Gets tokens per second for this sequence. |
| `UserContext` | Optional user context associated with this sequence. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AppendToken(Int32,Double)` | Appends a newly generated token to the sequence. |
| `Cancel` | Marks the sequence as cancelled. |
| `Complete(StopReason)` | Marks the sequence as complete. |
| `Fail(String)` | Marks the sequence as failed. |
| `ShouldStop(Int32,IReadOnlyCollection<Int32>)` | Checks if generation should stop based on various conditions. |

