---
title: "CommunicationManager"
description: "Central manager for distributed communication operations."
section: "API Reference"
---

`Helpers & Utilities` · `AiDotNet.DistributedTraining`

Central manager for distributed communication operations.

## For Beginners

This is your main entry point for distributed training communication.
It's a "wrapper" that makes it easy to communicate between different processes/GPUs
without worrying about the underlying implementation details.

## How It Works

**⚠️ WARNING - Static Mutable State:** This class uses static mutable state for managing
communication backends. This design choice has important implications for concurrent usage:

- Only ONE backend can be active per process
- Unit tests using this class CANNOT run in parallel
- Multiple training sessions in the same process share the same backend

See detailed thread-safety notes below for proper usage patterns.

Provides a static API for collective communication in distributed training scenarios.

Example usage:

IMPORTANT - Thread Safety and Testing Limitations:
This class uses STATIC MUTABLE STATE which has the following implications:

1. SINGLE GLOBAL INSTANCE: Only ONE backend can be active per process at a time.

Multiple training sessions in the same process will share the same backend instance.

2. PARALLEL TEST EXECUTION: Tests that use this class CANNOT run in parallel.

Use [Collection] attributes in xUnit or similar mechanisms to enforce sequential execution.

3. TEST ISOLATION: Always call Shutdown() in test cleanup to reset state.

For better isolation in tests, use InMemoryCommunicationBackend with unique environment IDs
and inject the backend directly instead of using this static manager.

4. CONCURRENT INITIALIZATION: Attempting to Initialize() from multiple threads concurrently

is protected by locks, but may result in exceptions if already initialized.

Recommended Test Pattern:

## Properties

| Property | Summary |
|:-----|:--------|
| `IsInitialized` | Gets whether the communication manager has been initialized. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AllGather(Vector<>)` | Gathers data from all processes and returns the concatenated result. |
| `AllReduce(Vector<>,ReductionOperation)` | Performs an AllReduce operation - combines data from all processes and distributes the result to all processes. |
| `Barrier` | Blocks until all processes reach this synchronization point. |
| `Broadcast(Vector<>,Int32)` | Broadcasts data from one process (root) to all other processes. |
| `GetBackend` | Gets the appropriate backend for the specified type. |
| `GetRank` | Gets the rank (ID) of the current process. |
| `GetWorldSize` | Gets the total number of processes in the distributed group. |
| `Initialize(ICommunicationBackend<>)` | Initializes the communication manager with the specified backend. |
| `ReduceScatter(Vector<>,ReductionOperation)` | Performs a reduce-scatter operation - combines data and distributes chunks. |
| `Scatter(Vector<>,Int32)` | Scatters different chunks of data from root to each process. |
| `Shutdown` | Shuts down the communication manager and releases all resources. |

