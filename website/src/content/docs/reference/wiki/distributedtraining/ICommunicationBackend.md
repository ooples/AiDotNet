---
title: "ICommunicationBackend<T>"
description: "Defines the contract for distributed communication backends."
section: "API Reference"
---

`Interfaces` · `AiDotNet.DistributedTraining`

Defines the contract for distributed communication backends.

## For Beginners

This interface defines how different processes (or GPUs) communicate with each other
during distributed training. Think of it as a "walkie-talkie" system where multiple
processes can send data to each other, synchronize, and perform collective operations.

## How It Works

This abstraction allows different implementations (in-memory, MPI.NET, NCCL, etc.)
to provide collective communication operations for distributed training.

## Properties

| Property | Summary |
|:-----|:--------|
| `IsInitialized` | Gets whether this backend is initialized and ready for use. |
| `Rank` | Gets the rank (ID) of the current process in the distributed group. |
| `WorldSize` | Gets the total number of processes in the distributed group. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AllGather(Vector<>)` | AllGather operation - gathers data from all processes and concatenates it. |
| `AllReduce(Vector<>,ReductionOperation)` | AllReduce operation - combines data from all processes using the specified operation and distributes the result back to all processes. |
| `Barrier` | Synchronization barrier - blocks until all processes reach this point. |
| `Broadcast(Vector<>,Int32)` | Broadcast operation - sends data from one process (root) to all other processes. |
| `Initialize` | Initializes the communication backend. |
| `Receive(Int32,Int32,Int32)` | Receive operation - receives data from a specific source process. |
| `ReduceScatter(Vector<>,ReductionOperation)` | ReduceScatter operation - reduces data and scatters the result. |
| `Scatter(Vector<>,Int32)` | Scatter operation - distributes different chunks of data from root to each process. |
| `Send(Vector<>,Int32,Int32)` | Send operation - sends data from this process to a specific destination process. |
| `Shutdown` | Shuts down the communication backend and releases resources. |

