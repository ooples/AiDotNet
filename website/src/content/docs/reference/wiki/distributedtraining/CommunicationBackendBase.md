---
title: "CommunicationBackendBase<T>"
description: "Provides base implementation for distributed communication backends."
section: "API Reference"
---

`Base Classes` · `AiDotNet.DistributedTraining`

Provides base implementation for distributed communication backends.

## For Beginners

This is the foundation that all communication systems build upon.

Think of this as a template that defines how any communication system should work.
It handles common tasks like:

- Keeping track of whether the system is initialized
- Validating inputs (checking for null values, correct sizes, etc.)
- Providing helper methods for common operations

Specific communication backends (like MPI or in-memory) inherit from this and add
their own implementation details. This prevents code duplication and ensures
all backends work consistently.

## How It Works

This abstract class implements common functionality for all communication backends,
including state management, validation, and helper methods for collective operations.
Derived classes implement the specific communication mechanisms (MPI, NCCL, in-memory, etc.).

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CommunicationBackendBase` | Initializes a new instance of the CommunicationBackendBase class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `IsInitialized` |  |
| `Rank` |  |
| `WorldSize` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `AllGather(Vector<>)` |  |
| `AllReduce(Vector<>,ReductionOperation)` |  |
| `ApplyReductionOperation(,,ReductionOperation)` | Applies a reduction operation to two values. |
| `Barrier` |  |
| `Broadcast(Vector<>,Int32)` |  |
| `EnsureInitialized` | Ensures the backend is initialized before performing operations. |
| `Initialize` |  |
| `OnInitialize` | Called during initialization to perform backend-specific setup. |
| `OnShutdown` | Called during shutdown to perform backend-specific cleanup. |
| `Receive(Int32,Int32,Int32)` |  |
| `ReduceScatter(Vector<>,ReductionOperation)` |  |
| `Scatter(Vector<>,Int32)` |  |
| `Send(Vector<>,Int32,Int32)` |  |
| `Shutdown` |  |
| `ValidateData(Vector<>,String)` | Validates that data is not null. |
| `ValidateRank(Int32,String)` | Validates that a rank is within valid bounds. |
| `ValidateRoot(Int32)` | Validates that a root rank is within valid bounds. |

## Fields

| Field | Summary |
|:-----|:--------|
| `NumOps` | Provides numeric operations for type T. |

