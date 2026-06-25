---
title: "InMemoryCommunicationBackend<T>"
description: "Provides an in-memory implementation of distributed communication for testing and single-machine scenarios."
section: "API Reference"
---

`Models & Types` · `AiDotNet.DistributedTraining`

Provides an in-memory implementation of distributed communication for testing and single-machine scenarios.

## For Beginners

This is a "fake" distributed system that runs on a single machine.

It's perfect for testing your distributed code without needing multiple GPUs or machines.
Think of it as a practice mode - it simulates distributed behavior but everything runs
in one process.

Use this when:

- Testing distributed code locally
- Debugging distributed training logic
- Running unit tests
- Learning how distributed training works

For production with actual multiple GPUs/machines, use an MPI-based backend instead.

Example:

## How It Works

**⚠️ WARNING - Static Shared State:**
This implementation uses STATIC shared dictionaries to simulate cross-process communication.
This design has important implications:

The static state includes: _sharedBuffers, _barrierCounters, _barrierGenerations, _operationCounters, _messageQueues.
These are namespaced by environmentId to enable concurrent independent sessions, but tests must ensure
unique environmentIds or run serially.

This backend simulates multiple processes by using shared memory and locks. It's perfect for testing
distributed code without needing actual MPI infrastructure or multiple machines. All "processes" run
within the same application instance, using static shared memory to simulate cross-process communication.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `InMemoryCommunicationBackend(Int32,Int32,String)` | Creates a new in-memory communication backend. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Rank` |  |
| `WorldSize` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `AllGather(Vector<>)` |  |
| `AllReduce(Vector<>,ReductionOperation)` | Performs an AllReduce operation, combining data from all processes. |
| `Barrier` |  |
| `Broadcast(Vector<>,Int32)` | Broadcasts data from the root process to all other processes. |
| `ClearEnvironment(String)` | Clears all shared state for a specific environment. |
| `OnInitialize` |  |
| `OnShutdown` |  |
| `PerformReduction(List<Vector<>>,ReductionOperation)` | Performs the actual reduction operation on a collection of vectors. |
| `Receive(Int32,Int32,Int32)` |  |
| `ReduceScatter(Vector<>,ReductionOperation)` |  |
| `Scatter(Vector<>,Int32)` |  |
| `Send(Vector<>,Int32,Int32)` |  |

