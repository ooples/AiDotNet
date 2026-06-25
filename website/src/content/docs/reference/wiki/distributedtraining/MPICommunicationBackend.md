---
title: "MPICommunicationBackend<T>"
description: "MPI.NET-based communication backend for production distributed training."
section: "API Reference"
---

`Models & Types` · `AiDotNet.DistributedTraining`

MPI.NET-based communication backend for production distributed training.

## How It Works

**Overview:**
MPI (Message Passing Interface) is the industry-standard communication framework for
high-performance computing. MPI.NET provides .NET bindings for MPI, enabling production-grade
distributed training on HPC clusters and supercomputers.

**Features:**

- Optimized collective operations (AllReduce, AllGather, etc.)
- Support for InfiniBand and other high-speed interconnects
- Battle-tested in HPC for decades
- Excellent performance and scalability

**Use Cases:**

- HPC cluster deployment
- Large-scale training (100s-1000s of nodes)
- InfiniBand or high-speed network infrastructure
- Production distributed training pipelines

**Requirements:**

- MPI.NET NuGet package
- MPI implementation (OpenMPI, MPICH, Intel MPI, etc.)
- MPI runtime environment

**Graceful Degradation:**
If MPI.NET is not available, this backend falls back to single-process mode
where all operations work correctly but without actual inter-process communication.
A warning is logged when fallback mode is active.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MPICommunicationBackend(Int32,Int32)` | Creates a new MPI communication backend. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Rank` |  |
| `WorldSize` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `AllGather(Vector<>)` |  |
| `AllReduce(Vector<>,ReductionOperation)` |  |
| `Barrier` |  |
| `Broadcast(Vector<>,Int32)` |  |
| `GetMPIOperation(ReductionOperation)` | Gets the MPI operation object for the specified reduction operation. |
| `OnInitialize` |  |
| `OnShutdown` |  |
| `Receive(Int32,Int32,Int32)` |  |
| `ReduceScatter(Vector<>,ReductionOperation)` |  |
| `Scatter(Vector<>,Int32)` |  |
| `Send(Vector<>,Int32,Int32)` |  |

